"""Shared context compaction helpers for all agents."""
from __future__ import annotations

import base64
import copy
import io
import json
import logging
import math
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..utils.llm_runtime import derive_bounded_stateful_session_id, invoke_with_retry

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


_DATA_URI_RE = re.compile(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)")

_COMPACT_KIND_KEY = "cytobridge_compact_kind"
_COMPACT_METADATA_KEY = "cytobridge_compact_metadata"
_COMPACT_BOUNDARY_KIND = "compact_boundary"
_COMPACT_SUMMARY_KIND = "compact_summary"
_COMPACT_REHYDRATE_KIND = "compact_rehydrate"
_MIN_CONTEXT_WINDOW = 1024
_MIN_TRIGGER_RATIO = 0.2
_MAX_TRIGGER_RATIO = 0.99
_DEFAULT_CONTEXT_WINDOW = 200000
_DEFAULT_CONTEXT_WINDOW_CAP = 256000

# Borrowed almost directly from Claude Code, then adapted to CytoBridge state.
NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

DETAILED_ANALYSIS_INSTRUCTION_BASE = """Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and workflow patterns
   - Specific details like:
     - file names
     - artifact paths
     - proposal, idea, and workspace identifiers
     - important config values
     - file edits and generated outputs
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly."""

BASE_COMPACT_PROMPT = f"""Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing workflow state, technical details, artifact bindings, and decisions that would be essential for continuing CytoBridge work without losing context.

{DETAILED_ANALYSIS_INSTRUCTION_BASE}

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, theories, data contracts, metrics, and frameworks discussed.
3. Files, Artifacts, and State: Enumerate specific files, artifacts, and state examined, modified, or created. Pay special attention to the most recent messages and include code/config snippets where applicable. Explicitly cover:
   - active workflow phase
   - task profile
   - current plan or current in-progress step
   - active data paths (`input_path`, `preprocessed_path`, `output_dir`)
   - active proposal / research idea / workspace / baseline run bindings
   - important downstream / report / training artifacts
   - pending `planner_need` / `runtime_action`
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the user's feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is directly in line with the user's most recent explicit requests and the task you were working on immediately before this summary request.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]

3. Files, Artifacts, and State:
   - [File or artifact]
      - [Why it matters]
      - [Important details, bindings, or snippets]

4. Errors and fixes:
    - [Detailed description of error]:
      - [How you fixed the error]
      - [User feedback on the error if any]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]

7. Pending Tasks:
   - [Task 1]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.
"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

CODEX_SUMMARY_PREFIX = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation."
)


def get_compact_prompt(custom_instructions: str = "") -> str:
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    if custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions.strip()}"
    prompt += NO_TOOLS_TRAILER
    return prompt


def format_compact_summary(summary: str) -> str:
    formatted = str(summary or "")
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted, count=1, flags=re.IGNORECASE)
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted, flags=re.IGNORECASE)
    if match:
        content = (match.group(1) or "").strip()
        formatted = f"Summary:\n{content}"
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def get_compact_user_summary_message(summary: str, *, recent_messages_preserved: bool = False) -> str:
    formatted = format_compact_summary(summary)
    base = f"{CODEX_SUMMARY_PREFIX}\n\n{formatted}"
    if recent_messages_preserved:
        base += "\n\nRecent messages are preserved verbatim."
    return base.strip()


def _count_text_tokens(text: str, enc: Any) -> int:
    if not text:
        return 0
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def _message_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "unknown"))
    return getattr(msg, "type", type(msg).__name__).replace("Message", "").lower()


def _message_content(msg: Any) -> Any:
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "") or ""


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        if "data:image/" in content:
            return _DATA_URI_RE.sub("[embedded image omitted]", content)
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            ptype = part.get("type")
            if ptype == "text":
                parts.append(str(part.get("text", "")))
            elif ptype == "image_url":
                parts.append("[image]")
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _safe_image_size_from_data_uri(url: str) -> Tuple[int, int]:
    if not isinstance(url, str) or not url.startswith("data:image/"):
        return 0, 0
    if Image is None:
        return 0, 0
    m = _DATA_URI_RE.match(url)
    if not m:
        return 0, 0
    try:
        payload = m.group(1)
        raw = base64.b64decode(payload, validate=False)
        with Image.open(io.BytesIO(raw)) as img:
            return int(img.width), int(img.height)
    except Exception:
        return 0, 0


def estimate_image_tokens(image_url: Any, detail: str = "auto") -> int:
    if isinstance(image_url, dict):
        url = image_url.get("url", "")
        detail = str(image_url.get("detail", detail) or detail)
    else:
        url = str(image_url or "")
        detail = str(detail or "auto")

    if detail == "low":
        return 85

    width, height = _safe_image_size_from_data_uri(url)
    if width > 0 and height > 0:
        tiles = max(1, math.ceil(width / 512) * math.ceil(height / 512))
        return min(85 + 170 * tiles, 4096)
    return 256


def _estimate_message_tokens(msg: Any, enc: Any) -> int:
    total = 4
    content = _message_content(msg)
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                total += _count_text_tokens(str(part), enc)
                continue
            ptype = part.get("type")
            if ptype == "text":
                total += _count_text_tokens(str(part.get("text", "")), enc)
            elif ptype == "image_url":
                total += estimate_image_tokens(part.get("image_url"))
            else:
                total += _count_text_tokens(str(part), enc)
    else:
        total += _count_text_tokens(_extract_text_from_content(content), enc)
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        total += _count_text_tokens(str(tool_calls), enc)
    return total


def estimate_tokens(messages: Sequence[Any], model: str = "gpt-4o") -> int:
    total = 0
    enc = None
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = None
    for msg in messages or []:
        total += _estimate_message_tokens(msg, enc)
    return total


def should_compact(total_tokens: int, context_window: int, ratio: float = 0.82) -> bool:
    threshold = int(max(1, context_window) * ratio)
    return total_tokens >= threshold


def _clone_message_with_content(msg: BaseMessage, new_content: Any) -> BaseMessage:
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update={"content": new_content})
    clone = copy.deepcopy(msg)
    clone.content = new_content
    return clone


def _clone_message_with_kwargs(msg: BaseMessage, **kwargs: Any) -> BaseMessage:
    if hasattr(msg, "model_copy"):
        return msg.model_copy(update=kwargs)
    clone = copy.deepcopy(msg)
    for key, value in kwargs.items():
        setattr(clone, key, value)
    return clone


def truncate_oversized_tool_messages(
    messages: Sequence[BaseMessage],
    max_chars: int = 1500,
) -> Tuple[List[BaseMessage], int]:
    truncated = 0
    out: List[BaseMessage] = []
    for msg in messages:
        content = _message_content(msg)
        if isinstance(msg, ToolMessage) and isinstance(content, str) and len(content) > max_chars:
            marker = f"\n...[truncated {len(content) - max_chars} chars]"
            out.append(_clone_message_with_content(msg, content[:max_chars] + marker))
            truncated += 1
        else:
            out.append(msg)
    return out, truncated


def _stringify_for_summary(msg: Any) -> str:
    role = _message_role(msg)
    text = _extract_text_from_content(_message_content(msg)).replace("\n", " ").strip()
    if len(text) > 220:
        text = text[:220] + "..."
    return f"- {role}: {text or '(empty)'}"


def _message_tool_call_ids(msg: Any) -> List[str]:
    if isinstance(msg, AIMessage):
        return [
            str(tc.get("id"))
            for tc in (getattr(msg, "tool_calls", None) or [])
            if isinstance(tc, dict) and tc.get("id")
        ]
    return []


def _expand_recent_boundary_for_tool_blocks(non_system: Sequence[Any], start_idx: int) -> int:
    start = max(0, int(start_idx))
    if start >= len(non_system):
        return len(non_system)
    if not isinstance(non_system[start], ToolMessage):
        return start

    block_start = start
    while block_start > 0 and isinstance(non_system[block_start - 1], ToolMessage):
        block_start -= 1
    if block_start == 0:
        return start

    candidate = non_system[block_start - 1]
    tool_ids = set(_message_tool_call_ids(candidate))
    if not tool_ids:
        return start

    block_ids = {
        str(getattr(non_system[idx], "tool_call_id", "") or "")
        for idx in range(block_start, len(non_system))
        if isinstance(non_system[idx], ToolMessage)
    }
    if tool_ids & block_ids:
        return block_start - 1
    return start


def _render_compaction_transcript(messages: Sequence[Any], max_chars: int) -> str:
    lines: List[str] = []
    for msg in messages:
        role = _message_role(msg)
        text = _extract_text_from_content(_message_content(msg)).strip()
        if len(text) > 900:
            text = text[:900] + " ...[truncated]"
        if not text:
            text = "(empty)"
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            try:
                names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict) and tc.get("name")]
                if names:
                    text += f" | tool_calls={','.join(names)}"
            except Exception:
                pass
        lines.append(f"[{role}] {text}")

    transcript = "\n".join(lines)
    if max_chars > 0 and len(transcript) > max_chars:
        head = max_chars // 2
        tail = max_chars - head
        transcript = transcript[:head] + "\n...[middle omitted]...\n" + transcript[-tail:]
    return transcript


def _compact_kind(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(((msg.get("additional_kwargs") or {}).get(_COMPACT_KIND_KEY)) or "")
    return str((getattr(msg, "additional_kwargs", {}) or {}).get(_COMPACT_KIND_KEY) or "")


def is_compact_boundary_message(msg: Any) -> bool:
    return _compact_kind(msg) == _COMPACT_BOUNDARY_KIND


def is_compact_summary_message(msg: Any) -> bool:
    return _compact_kind(msg) == _COMPACT_SUMMARY_KIND


def is_compact_rehydrate_message(msg: Any) -> bool:
    return _compact_kind(msg) == _COMPACT_REHYDRATE_KIND


def _compact_metadata(msg: Any) -> Dict[str, Any]:
    if isinstance(msg, dict):
        return dict(((msg.get("additional_kwargs") or {}).get(_COMPACT_METADATA_KEY)) or {})
    return dict((getattr(msg, "additional_kwargs", {}) or {}).get(_COMPACT_METADATA_KEY) or {})


def find_last_compact_boundary_index(messages: Sequence[Any]) -> int:
    seq = list(messages or [])
    for idx in range(len(seq) - 1, -1, -1):
        if is_compact_boundary_message(seq[idx]):
            return idx
    return -1


def _deserialize_message_dump(raw: Any) -> Optional[BaseMessage]:
    try:
        loaded = load(raw)
    except Exception:
        return None
    return loaded if isinstance(loaded, BaseMessage) else None


def _deserialize_preserved_tail(boundary: Any) -> List[BaseMessage]:
    metadata = _compact_metadata(boundary)
    preserved: List[BaseMessage] = []
    for item in metadata.get("preserved_tail") or []:
        loaded = _deserialize_message_dump(item)
        if loaded is not None:
            preserved.append(loaded)
    return preserved


def project_compacted_history(
    messages: Sequence[BaseMessage],
    *,
    preserve_leading_system: bool = False,
) -> Tuple[List[BaseMessage], Dict[str, Any]]:
    seq = list(messages or [])
    if not seq:
        return [], {"has_boundary": False}

    leading: List[BaseMessage] = []
    body = seq
    if preserve_leading_system and isinstance(seq[0], SystemMessage) and not is_compact_boundary_message(seq[0]):
        leading = [seq[0]]
        body = seq[1:]

    boundary_idx = find_last_compact_boundary_index(body)
    if boundary_idx < 0:
        return list(seq), {"has_boundary": False}

    boundary = body[boundary_idx]
    after_boundary = body[boundary_idx + 1 :]
    synthetic: List[BaseMessage] = []
    rest_start = 0
    for idx, message in enumerate(after_boundary):
        if is_compact_summary_message(message) or is_compact_rehydrate_message(message):
            synthetic.append(message)
            rest_start = idx + 1
            continue
        break
    rest = after_boundary[rest_start:]
    preserved_tail = _deserialize_preserved_tail(boundary)
    projected = leading + [boundary] + synthetic + preserved_tail + rest
    return projected, {
        "has_boundary": True,
        "boundary_index": boundary_idx + len(leading),
        "preserved_tail_count": len(preserved_tail),
        "synthetic_count": len(synthetic),
    }


def _message_has_synthetic_media(msg: BaseMessage) -> bool:
    if not isinstance(msg, HumanMessage):
        return False
    additional = dict(getattr(msg, "additional_kwargs", {}) or {})
    return bool(additional.get("synthetic_tool_media"))


def _message_is_already_microcompacted(msg: BaseMessage) -> bool:
    additional = dict(getattr(msg, "additional_kwargs", {}) or {})
    if bool(additional.get("tool_message_microcompacted")):
        return True
    content = getattr(msg, "content", None)
    return isinstance(content, str) and "...[microcompacted " in content


def microcompact_history(
    messages: Sequence[BaseMessage],
    *,
    keep_last_turns: int = 6,
    max_tool_chars: int = 1500,
    preserve_leading_system: bool = False,
) -> Tuple[List[BaseMessage], Dict[str, Any]]:
    seq = list(messages or [])
    if not seq:
        return [], {
            "mode": "none",
            "compacted_tool_messages": 0,
            "cleared_media_messages": 0,
            "tokens_saved": 0,
        }

    leading: List[BaseMessage] = []
    body = seq
    if preserve_leading_system and isinstance(seq[0], SystemMessage) and not is_compact_boundary_message(seq[0]):
        leading = [seq[0]]
        body = seq[1:]

    non_system = [msg for msg in body if not isinstance(msg, SystemMessage)]
    if not non_system:
        return seq, {
            "mode": "none",
            "compacted_tool_messages": 0,
            "cleared_media_messages": 0,
            "tokens_saved": 0,
        }

    keep_n = max(2, keep_last_turns * 2)
    start_idx = max(0, len(non_system) - keep_n)
    start_idx = _expand_recent_boundary_for_tool_blocks(non_system, start_idx)
    recent_ids = {id(msg) for msg in non_system[start_idx:]}

    before_tokens = estimate_tokens(seq)
    compacted_tool_messages = 0
    cleared_media_messages = 0
    rewritten_body: List[BaseMessage] = []
    tool_char_limit = max(200, int(max_tool_chars or 1500))

    for msg in body:
        if id(msg) in recent_ids:
            rewritten_body.append(msg)
            continue
        if (
            isinstance(msg, ToolMessage)
            and isinstance(msg.content, str)
            and len(msg.content) > tool_char_limit
            and not _message_is_already_microcompacted(msg)
        ):
            marker = f"\n...[microcompacted {len(msg.content) - tool_char_limit} chars]"
            trimmed = _clone_message_with_content(msg, msg.content[:tool_char_limit] + marker)
            rewritten_body.append(
                _clone_message_with_kwargs(
                    trimmed,
                    additional_kwargs={
                        **dict(getattr(trimmed, "additional_kwargs", {}) or {}),
                        "tool_message_microcompacted": True,
                    },
                )
            )
            compacted_tool_messages += 1
            continue
        if _message_has_synthetic_media(msg):
            replacement = _clone_message_with_content(msg, "[older synthetic tool media omitted during microcompact]")
            rewritten_body.append(
                _clone_message_with_kwargs(
                    replacement,
                    additional_kwargs={
                        **dict(getattr(replacement, "additional_kwargs", {}) or {}),
                        "synthetic_tool_media_microcompacted": True,
                    },
                )
            )
            cleared_media_messages += 1
            continue
        rewritten_body.append(msg)

    rewritten = leading + rewritten_body
    after_tokens = estimate_tokens(rewritten)
    changed = compacted_tool_messages > 0 or cleared_media_messages > 0
    return rewritten, {
        "mode": "microcompact" if changed else "none",
        "compacted_tool_messages": compacted_tool_messages,
        "cleared_media_messages": cleared_media_messages,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
    }


def _extract_summary_from_llm_response(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    return format_compact_summary(raw) or raw


def _build_isolated_compaction_llm(llm: Any) -> Any:
    if llm is None:
        return None

    clone = llm
    try:
        if hasattr(llm, "model_copy"):
            clone = llm.model_copy(deep=False)
        elif hasattr(llm, "copy"):
            clone = llm.copy(deep=False)
    except Exception:
        clone = llm

    if clone is not llm and hasattr(clone, "session_id"):
        base_session = str(getattr(llm, "session_id", "") or "cytobridge")
        setattr(
            clone,
            "session_id",
            derive_bounded_stateful_session_id(
                base_session,
                f"-compaction-{uuid.uuid4().hex[:8]}",
            ),
        )
    return clone


def _render_cytobridge_compact_instructions(state: Optional[Dict[str, Any]]) -> str:
    st = dict(state or {})
    task_profile = dict(st.get("task_profile") or {})
    final_config = dict(st.get("final_config") or {})
    lines = [
        "When filling the 'Files, Artifacts, and State' section, explicitly summarize:",
        f"- workflow_phase: {st.get('workflow_phase') or 'intake'}",
        f"- planner_phase: {st.get('planner_phase') or 'working'}",
        f"- task_profile: {json.dumps(task_profile, ensure_ascii=False, sort_keys=True)}",
        f"- current_plan_or_decision: {json.dumps(st.get('plan_decision') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- input_path: {st.get('input_path') or '(none)'}",
        f"- preprocessed_path: {st.get('preprocessed_path') or '(none)'}",
        f"- output_dir: {st.get('output_dir') or '(none)'}",
        f"- final_model_path: {final_config.get('path') or '(none)'}",
        f"- active_proposal_id: {st.get('active_proposal_id') or '(none)'}",
        f"- active_research_idea_id: {st.get('active_research_idea_id') or '(none)'}",
        f"- active_workspace_snapshot_id: {st.get('active_workspace_snapshot_id') or '(none)'}",
        f"- active_baseline_run_id: {st.get('active_baseline_run_id') or '(none)'}",
        f"- planner_need: {json.dumps(st.get('planner_need') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- runtime_action: {json.dumps(st.get('runtime_action') or {}, ensure_ascii=False, sort_keys=True)}",
    ]
    return "\n".join(lines)


def _llm_compact_summary(
    old_messages: Sequence[BaseMessage],
    *,
    llm: Any,
    state: Optional[Dict[str, Any]] = None,
    max_input_chars: int = 24000,
) -> str:
    if llm is None or not old_messages:
        return ""
    transcript = _render_compaction_transcript(old_messages, max_chars=max_input_chars)
    if not transcript.strip():
        return ""

    prompt = [
        SystemMessage(content=get_compact_prompt(_render_cytobridge_compact_instructions(state))),
        HumanMessage(
            content=(
                "Conversation transcript to summarize:\n\n"
                f"{transcript}"
            )
        ),
    ]
    compaction_llm = _build_isolated_compaction_llm(llm)
    try:
        response = invoke_with_retry(compaction_llm, prompt, logger, "compaction")
    except Exception:
        logger.debug("LLM compaction failed; falling back to rule summary", exc_info=True)
        return ""
    raw = _extract_text_from_content(getattr(response, "content", ""))
    return _extract_summary_from_llm_response(raw)


def _render_task_profile_summary(state: Dict[str, Any]) -> str:
    profile = dict(state.get("task_profile") or {})
    facets = [key for key, value in (profile.get("facets") or {}).items() if value]
    axes = [key for key, value in (profile.get("change_axes") or {}).items() if value]
    risks = [key for key, value in (profile.get("risk_flags") or {}).items() if value]
    return "\n".join(
        [
            f"- current_stage: {profile.get('current_stage') or state.get('workflow_phase') or 'intake'}",
            f"- primary_goal: {profile.get('primary_goal') or 'analysis'}",
            f"- active_facets: {', '.join(facets) if facets else '(none)'}",
            f"- active_change_axes: {', '.join(axes) if axes else '(none)'}",
            f"- active_risk_flags: {', '.join(risks) if risks else '(none)'}",
            f"- notes: {str(profile.get('notes') or '').strip() or '(none)'}",
        ]
    )


def _render_artifact_inventory(state: Dict[str, Any]) -> str:
    final_config = dict(state.get("final_config") or {})
    training_runs = list(state.get("training_runs") or [])
    downstream_results = list(state.get("downstream_results") or [])
    downstream_figures = list(state.get("downstream_figures") or [])
    lines = [
        f"- input_path: {state.get('input_path') or '(none)'}",
        f"- preprocessed_path: {state.get('preprocessed_path') or '(none)'}",
        f"- output_dir: {state.get('output_dir') or '(none)'}",
        f"- final_model_path: {final_config.get('path') or '(none)'}",
        f"- active_proposal_id: {state.get('active_proposal_id') or '(none)'}",
        f"- active_research_idea_id: {state.get('active_research_idea_id') or '(none)'}",
        f"- active_workspace_snapshot_id: {state.get('active_workspace_snapshot_id') or '(none)'}",
        f"- active_baseline_run_id: {state.get('active_baseline_run_id') or '(none)'}",
    ]
    if training_runs:
        recent_run = training_runs[-1]
        lines.append(f"- latest_training_run: {json.dumps(recent_run, ensure_ascii=False, sort_keys=True)}")
    if downstream_results:
        lines.append(f"- latest_downstream_result: {json.dumps(downstream_results[-1], ensure_ascii=False, sort_keys=True)}")
    if downstream_figures:
        lines.append(f"- latest_downstream_figure: {json.dumps(downstream_figures[-1], ensure_ascii=False, sort_keys=True)}")
    return "\n".join(lines)


def build_compaction_rehydrate_messages(state: Optional[Dict[str, Any]]) -> List[SystemMessage]:
    st = dict(state or {})
    plan_decision = dict(st.get("plan_decision") or {})
    planner_need = dict(st.get("planner_need") or {})
    runtime_action = dict(st.get("runtime_action") or {})
    lines = [
        "Compaction rehydrate state:",
        "",
        "Workflow and planner state:",
        f"- workflow_phase: {st.get('workflow_phase') or 'intake'}",
        f"- planner_phase: {st.get('planner_phase') or 'working'}",
        "",
        "Task profile:",
        _render_task_profile_summary(st),
        "",
        "Plan and pending actions:",
        f"- plan_decision: {json.dumps(plan_decision, ensure_ascii=False, sort_keys=True)}",
        f"- planner_need: {json.dumps(planner_need, ensure_ascii=False, sort_keys=True)}",
        f"- runtime_action: {json.dumps(runtime_action, ensure_ascii=False, sort_keys=True)}",
        "",
        "Artifacts and bindings:",
        _render_artifact_inventory(st),
    ]
    return [
        SystemMessage(
            content="\n".join(lines).strip(),
            additional_kwargs={
                _COMPACT_KIND_KEY: _COMPACT_REHYDRATE_KIND,
                _COMPACT_METADATA_KEY: {
                    "created_at": _utc_now_iso(),
                    "workflow_phase": st.get("workflow_phase") or "intake",
                },
            },
        )
    ]


def _make_boundary_message(
    *,
    trigger: str,
    pre_tokens: int,
    messages_summarized: int,
    summary_mode: str,
    preserved_tail: Sequence[BaseMessage],
    microcompact_meta: Optional[Dict[str, Any]] = None,
) -> SystemMessage:
    return SystemMessage(
        content="Conversation compacted",
        additional_kwargs={
            _COMPACT_KIND_KEY: _COMPACT_BOUNDARY_KIND,
            _COMPACT_METADATA_KEY: {
                "trigger": str(trigger or "auto"),
                "pre_tokens": int(pre_tokens),
                "messages_summarized": int(messages_summarized),
                "summary_mode": str(summary_mode or "rule"),
                "preserved_tail": [dumpd(msg) for msg in preserved_tail],
                "microcompact": dict(microcompact_meta or {}),
                "created_at": _utc_now_iso(),
            },
        },
    )


def _make_summary_message(summary_text: str) -> SystemMessage:
    return SystemMessage(
        content=str(summary_text or "").strip(),
        additional_kwargs={
            _COMPACT_KIND_KEY: _COMPACT_SUMMARY_KIND,
            _COMPACT_METADATA_KEY: {
                "created_at": _utc_now_iso(),
            },
        },
    )


def _truncate_compact_text(text: str, max_chars: int) -> str:
    raw = str(text or "").strip()
    limit = max(0, int(max_chars or 0))
    if limit <= 0 or len(raw) <= limit:
        return raw
    marker = "\n...[compact summary truncated]..."
    if limit <= len(marker) + 32:
        return (raw[:limit] + "...")[:limit]
    head = max(64, int(limit * 0.7))
    tail = max(16, limit - head - len(marker))
    return (raw[:head] + marker + raw[-tail:]).strip()


def _materialize_compacted_view(
    *,
    prefix: Sequence[BaseMessage],
    trigger: str,
    pre_tokens: int,
    old_messages: Sequence[BaseMessage],
    mode: str,
    preserved_tail: Sequence[BaseMessage],
    summary_text: str,
    state: Optional[Dict[str, Any]],
    microcompact_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List[BaseMessage], List[BaseMessage]]:
    boundary_msg = _make_boundary_message(
        trigger=trigger,
        pre_tokens=pre_tokens,
        messages_summarized=len(old_messages),
        summary_mode=mode,
        preserved_tail=preserved_tail,
        microcompact_meta=microcompact_meta,
    )
    summary_msg = _make_summary_message(summary_text)
    rehydrate_messages = build_compaction_rehydrate_messages(state)
    history_updates: List[BaseMessage] = [boundary_msg, summary_msg, *rehydrate_messages]
    compacted_view = list(prefix) + history_updates + list(preserved_tail)
    return compacted_view, history_updates


def _legacy_rule_compact_history(
    messages: Sequence[Any],
    *,
    keep_last_turns: int = 6,
    llm: Any = None,
    llm_enabled: bool = True,
    llm_input_max_chars: int = 24000,
    short_history_max_tool_chars: int = 1500,
) -> Tuple[List[Any], Dict[str, Any]]:
    seq = list(messages or [])
    if not seq:
        return [], {"removed_messages": 0, "summary_added": False, "mode": "none"}
    if len(seq) <= (keep_last_turns * 2 + 2):
        return seq, {"removed_messages": 0, "summary_added": False, "mode": "none"}
    system_msgs: List[Any] = []
    non_system: List[Any] = []
    for msg in seq:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
        if role in {"system", "SystemMessage"}:
            system_msgs.append(msg)
        else:
            non_system.append(msg)
    keep_n = max(2, keep_last_turns * 2)
    recent = non_system[-keep_n:]
    old = non_system[:-keep_n]
    summary_core = ""
    if llm_enabled and llm is not None:
        rendered = _render_compaction_transcript(old, max_chars=max(2000, int(llm_input_max_chars or 24000)))
        if rendered:
            summary_core = rendered
    if not summary_core:
        summary_core = "\n".join(_stringify_for_summary(msg) for msg in old[-60:])
    summary_text = get_compact_user_summary_message(summary_core, recent_messages_preserved=True)
    summary_msg: Any
    if isinstance(seq[0], dict):
        summary_msg = {"role": "system", "content": summary_text}
    else:
        summary_msg = _make_summary_message(summary_text)
    compacted = system_msgs[:1] + [summary_msg] + recent
    return compacted, {
        "removed_messages": len(old),
        "summary_added": True,
        "summary_chars": len(summary_text),
        "summary_text": summary_text,
        "summary_core": summary_core,
        "mode": "legacy_rule",
        "history_updates": [],
        "before_tokens": estimate_tokens(seq),
        "after_tokens": estimate_tokens(compacted),
        "projection_mode": "legacy",
    }


def _select_recent_user_messages_for_tail(
    messages: Sequence[BaseMessage],
    *,
    max_messages: int,
) -> List[BaseMessage]:
    limit = max(0, int(max_messages or 0))
    if limit <= 0:
        return []
    selected: List[BaseMessage] = []
    for msg in reversed(list(messages or [])):
        if isinstance(msg, HumanMessage):
            selected.append(msg)
            if len(selected) >= limit:
                break
    selected.reverse()
    return selected


def compact_history(
    messages: Sequence[Any],
    keep_last_turns: int = 6,
    summary_prefix: str = CODEX_SUMMARY_PREFIX,  # retained for compatibility; no longer used directly
    llm: Any = None,
    llm_enabled: bool = True,
    llm_input_max_chars: int = 24000,
    short_history_max_tool_chars: int = 1500,
    *,
    preserve_leading_system: bool = False,
    state: Optional[Dict[str, Any]] = None,
    trigger: str = "auto",
    force_full: bool = True,
    microcompact_enabled: bool = True,
    context_window: Optional[int] = None,
    trigger_ratio: float = 0.82,
) -> Tuple[List[Any], Dict[str, Any]]:
    del summary_prefix
    seq = list(messages or [])
    if not seq:
        return [], {
            "removed_messages": 0,
            "summary_added": False,
            "mode": "none",
            "history_updates": [],
            "before_tokens": 0,
            "after_tokens": 0,
            "projection_mode": "none",
        }

    if seq and isinstance(seq[0], dict):
        return _legacy_rule_compact_history(
            seq,
            keep_last_turns=keep_last_turns,
            llm=llm,
            llm_enabled=llm_enabled,
            llm_input_max_chars=llm_input_max_chars,
            short_history_max_tool_chars=short_history_max_tool_chars,
        )

    base_projection, projection_meta = project_compacted_history(
        seq,
        preserve_leading_system=preserve_leading_system,
    )
    before_tokens = estimate_tokens(base_projection)
    micro_meta = {
        "mode": "none",
        "compacted_tool_messages": 0,
        "cleared_media_messages": 0,
        "tokens_saved": 0,
    }
    working_projection = list(base_projection)
    if microcompact_enabled:
        working_projection, micro_meta = microcompact_history(
            base_projection,
            keep_last_turns=keep_last_turns,
            max_tool_chars=short_history_max_tool_chars,
            preserve_leading_system=preserve_leading_system,
        )
    after_micro_tokens = estimate_tokens(working_projection)
    if context_window is not None and not force_full:
        if not should_compact(after_micro_tokens, int(context_window), float(trigger_ratio)):
            mode = micro_meta.get("mode", "none")
            return working_projection, {
                "removed_messages": 0,
                "summary_added": False,
                "summary_chars": 0,
                "summary_text": "",
                "summary_core": "",
                "mode": mode,
                "history_updates": [],
                "before_tokens": before_tokens,
                "after_tokens": after_micro_tokens,
                "microcompact": dict(micro_meta),
                "projection_mode": "boundary" if projection_meta.get("has_boundary") else "full",
                "projection_meta": dict(projection_meta),
            }

    prefix: List[BaseMessage] = []
    body = working_projection
    if preserve_leading_system and isinstance(working_projection[0], SystemMessage) and not is_compact_boundary_message(working_projection[0]):
        prefix = [working_projection[0]]
        body = working_projection[1:]

    non_system = [msg for msg in body if not isinstance(msg, SystemMessage)]
    if not non_system:
        mode = micro_meta.get("mode", "none")
        return working_projection, {
            "removed_messages": 0,
            "summary_added": False,
            "summary_chars": 0,
            "summary_text": "",
            "summary_core": "",
            "mode": mode,
            "history_updates": [],
            "before_tokens": before_tokens,
            "after_tokens": after_micro_tokens,
            "microcompact": dict(micro_meta),
            "projection_mode": "boundary" if projection_meta.get("has_boundary") else "full",
            "projection_meta": dict(projection_meta),
        }

    old_messages = list(body)
    preserved_tail = _select_recent_user_messages_for_tail(
        old_messages,
        max_messages=keep_last_turns,
    )

    summary_core = ""
    mode = "rule"
    if llm_enabled and llm is not None:
        summary_core = _llm_compact_summary(
            old_messages,
            llm=llm,
            state=state,
            max_input_chars=max(2000, int(llm_input_max_chars or 24000)),
        )
        if summary_core:
            mode = "llm"
    if not summary_core:
        summary_core = "\n".join(_stringify_for_summary(msg) for msg in old_messages[-80:])
        mode = "rule"

    summary_text = get_compact_user_summary_message(
        summary_core,
        recent_messages_preserved=bool(preserved_tail),
    )
    current_tail: List[BaseMessage] = list(preserved_tail)
    current_summary_text = summary_text
    compacted_view, history_updates = _materialize_compacted_view(
        prefix=prefix,
        trigger=trigger,
        pre_tokens=before_tokens,
        old_messages=old_messages,
        mode=mode,
        preserved_tail=current_tail,
        summary_text=current_summary_text,
        state=state,
        microcompact_meta=micro_meta,
    )
    after_tokens = estimate_tokens(compacted_view)

    summary_truncated = False
    if context_window is not None:
        threshold = int(max(1, int(context_window)) * float(trigger_ratio))
        target_tokens = max(128, min(threshold - 128, int(threshold * 0.8)))

        while current_tail and after_tokens > target_tokens:
            current_tail = current_tail[1:]
            if not current_tail:
                current_summary_text = get_compact_user_summary_message(
                    summary_core,
                    recent_messages_preserved=False,
                )
            compacted_view, history_updates = _materialize_compacted_view(
                prefix=prefix,
                trigger=trigger,
                pre_tokens=before_tokens,
                old_messages=old_messages,
                mode=mode,
                preserved_tail=current_tail,
                summary_text=current_summary_text,
                state=state,
                microcompact_meta=micro_meta,
            )
            after_tokens = estimate_tokens(compacted_view)

        if after_tokens > target_tokens:
            fixed_view, _ = _materialize_compacted_view(
                prefix=prefix,
                trigger=trigger,
                pre_tokens=before_tokens,
                old_messages=old_messages,
                mode=mode,
                preserved_tail=current_tail,
                summary_text="",
                state=state,
                microcompact_meta=micro_meta,
            )
            fixed_tokens = estimate_tokens(fixed_view)
            summary_budget_tokens = max(96, target_tokens - fixed_tokens)
            summary_budget_chars = max(384, summary_budget_tokens * 4)
            truncated_summary = _truncate_compact_text(current_summary_text, summary_budget_chars)
            if truncated_summary != current_summary_text:
                current_summary_text = truncated_summary
                summary_truncated = True
                compacted_view, history_updates = _materialize_compacted_view(
                    prefix=prefix,
                    trigger=trigger,
                    pre_tokens=before_tokens,
                    old_messages=old_messages,
                    mode=mode,
                    preserved_tail=current_tail,
                    summary_text=current_summary_text,
                    state=state,
                    microcompact_meta=micro_meta,
                )
                after_tokens = estimate_tokens(compacted_view)

    return compacted_view, {
        "removed_messages": len(old_messages),
        "summary_added": True,
        "summary_chars": len(current_summary_text),
        "summary_text": current_summary_text,
        "summary_core": summary_core,
        "mode": mode,
        "history_updates": history_updates,
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "microcompact": dict(micro_meta),
        "projection_mode": "boundary" if projection_meta.get("has_boundary") else "full",
        "projection_meta": dict(projection_meta),
        "preserved_tail_count": len(current_tail),
        "tail_pruned": 0,
        "summary_truncated": bool(summary_truncated),
    }


def is_context_overflow_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "context length",
        "maximum context",
        "too many tokens",
        "context window",
        "reduce the length",
        "input is too long",
        "maximum number of tokens",
    ]
    return any(marker in text for marker in markers)


def record_compaction_stats(
    stats: Optional[Dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
    reason: str,
    *,
    mode: str = "",
) -> Dict[str, Any]:
    base = dict(stats or {})
    base["count"] = int(base.get("count", 0)) + 1
    base["last_at"] = _utc_now_iso()
    base["last_before_tokens"] = int(before_tokens)
    base["last_after_tokens"] = int(after_tokens)
    base["last_reason"] = str(reason)
    if mode:
        base["last_mode"] = str(mode)
    return base


def record_microcompact_stats(
    stats: Optional[Dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
) -> Dict[str, Any]:
    base = dict(stats or {})
    base["microcompact_count"] = int(base.get("microcompact_count", 0)) + 1
    base["last_microcompact_at"] = _utc_now_iso()
    base["last_microcompact_before_tokens"] = int(before_tokens)
    base["last_microcompact_after_tokens"] = int(after_tokens)
    return base


def get_context_policy(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = {
        "enabled": True,
        "context_window": _DEFAULT_CONTEXT_WINDOW,
        "context_window_cap": _DEFAULT_CONTEXT_WINDOW_CAP,
        "effective_context_window": _DEFAULT_CONTEXT_WINDOW,
        "allow_large_context_window": False,
        "trigger_ratio": 0.82,
        "keep_last_turns": 6,
        "max_tool_chars": 1500,
        "microcompact_enabled": True,
        "llm_compact_enabled": True,
        "llm_compact_input_max_chars": 24000,
    }
    if overrides:
        policy.update(overrides)
    try:
        policy["context_window"] = int(policy.get("context_window", _DEFAULT_CONTEXT_WINDOW))
    except Exception:
        policy["context_window"] = _DEFAULT_CONTEXT_WINDOW
    if policy["context_window"] < _MIN_CONTEXT_WINDOW:
        policy["context_window"] = _DEFAULT_CONTEXT_WINDOW

    try:
        policy["context_window_cap"] = int(policy.get("context_window_cap", _DEFAULT_CONTEXT_WINDOW_CAP))
    except Exception:
        policy["context_window_cap"] = _DEFAULT_CONTEXT_WINDOW_CAP
    if policy["context_window_cap"] < _MIN_CONTEXT_WINDOW:
        policy["context_window_cap"] = _DEFAULT_CONTEXT_WINDOW_CAP

    policy["allow_large_context_window"] = bool(policy.get("allow_large_context_window", False))
    if policy["allow_large_context_window"]:
        policy["effective_context_window"] = int(policy["context_window"])
    else:
        policy["effective_context_window"] = int(min(int(policy["context_window"]), int(policy["context_window_cap"])))

    try:
        policy["trigger_ratio"] = float(policy.get("trigger_ratio", 0.82))
    except Exception:
        policy["trigger_ratio"] = 0.82
    if not (_MIN_TRIGGER_RATIO <= policy["trigger_ratio"] < _MAX_TRIGGER_RATIO):
        policy["trigger_ratio"] = 0.82

    try:
        policy["keep_last_turns"] = max(1, int(policy.get("keep_last_turns", 6)))
    except Exception:
        policy["keep_last_turns"] = 6

    try:
        policy["max_tool_chars"] = max(128, int(policy.get("max_tool_chars", 1500)))
    except Exception:
        policy["max_tool_chars"] = 1500

    try:
        policy["llm_compact_input_max_chars"] = max(2000, int(policy.get("llm_compact_input_max_chars", 24000)))
    except Exception:
        policy["llm_compact_input_max_chars"] = 24000

    policy["enabled"] = bool(policy.get("enabled", True))
    policy["microcompact_enabled"] = bool(policy.get("microcompact_enabled", True))
    policy["llm_compact_enabled"] = bool(policy.get("llm_compact_enabled", True))
    return policy
