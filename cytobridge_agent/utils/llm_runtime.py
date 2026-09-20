from __future__ import annotations

import logging
import os
import random
import time
import uuid
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage


RATE_LIMIT_FIXED_RETRY_SEC = 60.0
STATEFUL_SESSION_ID_MAX_LEN = 64


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def get_llm_runtime_settings() -> Dict[str, Any]:
    """Centralized runtime controls for provider calls."""
    return {
        "max_retries": _env_int("CYTOBRIDGE_LLM_MAX_RETRIES", default=3, min_value=0, max_value=20),
        "sdk_max_retries": _env_int("CYTOBRIDGE_LLM_SDK_MAX_RETRIES", default=0, min_value=0, max_value=10),
        "timeout_sec": _env_float("CYTOBRIDGE_LLM_TIMEOUT_SEC", default=300.0, min_value=5.0, max_value=1800.0),
        "retry_backoff_sec": _env_float("CYTOBRIDGE_LLM_RETRY_BACKOFF_SEC", default=1.2, min_value=0.1, max_value=30.0),
        "retry_jitter_sec": _env_float("CYTOBRIDGE_LLM_RETRY_JITTER_SEC", default=0.3, min_value=0.0, max_value=10.0),
        "rate_limit_retry_sec": _env_float("CYTOBRIDGE_LLM_RATE_LIMIT_RETRY_SEC", default=RATE_LIMIT_FIXED_RETRY_SEC, min_value=1.0, max_value=600.0),
        "retry_all_errors": os.getenv("CYTOBRIDGE_LLM_RETRY_ALL_ERRORS", "1").strip() not in {"0", "false", "False"},
        "semantic_retry_max": _env_int("CYTOBRIDGE_LLM_SEMANTIC_RETRIES", default=2, min_value=0, max_value=10),
    }


def extract_usage_stats(message: Any) -> Dict[str, Optional[float]]:
    usage_meta = getattr(message, "usage_metadata", None)
    usage_meta = usage_meta if isinstance(usage_meta, dict) else {}

    response_meta = getattr(message, "response_metadata", None)
    response_meta = response_meta if isinstance(response_meta, dict) else {}

    token_usage = response_meta.get("token_usage")
    token_usage = token_usage if isinstance(token_usage, dict) else {}

    prompt_tokens = _first_present(
        usage_meta.get("input_tokens"),
        usage_meta.get("prompt_tokens"),
        token_usage.get("prompt_tokens"),
    )
    completion_tokens = _first_present(
        usage_meta.get("output_tokens"),
        usage_meta.get("completion_tokens"),
        token_usage.get("completion_tokens"),
    )
    total_tokens = _first_present(
        usage_meta.get("total_tokens"),
        token_usage.get("total_tokens"),
    )

    cached_tokens = None
    prompt_details = token_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")

    if cached_tokens is None:
        input_details = usage_meta.get("input_token_details")
        if isinstance(input_details, dict):
            cached_tokens = input_details.get("cache_read")
            if cached_tokens is None:
                cached_tokens = input_details.get("cached_tokens")

    cache_hit_rate = None
    try:
        if prompt_tokens and cached_tokens is not None and int(prompt_tokens) > 0:
            cache_hit_rate = float(cached_tokens) / float(prompt_tokens)
    except Exception:
        cache_hit_rate = None

    return {
        "prompt_tokens": float(prompt_tokens) if prompt_tokens is not None else None,
        "completion_tokens": float(completion_tokens) if completion_tokens is not None else None,
        "total_tokens": float(total_tokens) if total_tokens is not None else None,
        "cached_prompt_tokens": float(cached_tokens) if cached_tokens is not None else None,
        "cache_hit_rate": cache_hit_rate,
    }


def extract_response_debug(message: Any) -> Dict[str, Any]:
    content = getattr(message, "content", "")
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    tool_calls = list(getattr(message, "tool_calls", []) or [])
    response_metadata = dict(getattr(message, "response_metadata", {}) or {})

    if isinstance(content, list):
        try:
            rendered_content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
        except Exception:
            rendered_content = str(content)
    else:
        rendered_content = str(content or "")

    return {
        "response_text": rendered_content,
        "response_tool_calls": tool_calls,
        "response_phase": response_metadata.get("phase") or additional_kwargs.get("phase"),
        "response_finish_reason": response_metadata.get("finish_reason"),
        "response_error_message": response_metadata.get("error_message"),
    }


def log_llm_response_stats(logger: logging.Logger, agent: str, message: Any) -> None:
    stats = extract_usage_stats(message)
    debug = extract_response_debug(message)

    prompt = stats["prompt_tokens"]
    completion = stats["completion_tokens"]
    total = stats["total_tokens"]
    cached = stats["cached_prompt_tokens"]
    hit = stats["cache_hit_rate"]

    if prompt is None and completion is None and total is None:
        logger.debug("[%s] LLM usage metadata unavailable", agent)
        return

    hit_str = "n/a" if hit is None else f"{hit * 100:.2f}%"
    logger.info(
        "[%s] LLM usage prompt=%s completion=%s total=%s cached_prompt=%s cache_hit=%s",
        agent,
        int(prompt) if prompt is not None else "n/a",
        int(completion) if completion is not None else "n/a",
        int(total) if total is not None else "n/a",
        int(cached) if cached is not None else "n/a",
        hit_str,
    )

    # Also emit to Web UI timeline when event callback is registered.
    try:
        from ..display import DisplayManager

        DisplayManager()._emit(
            "llm_usage",
            {
                "agent": agent,
                "prompt_tokens": int(prompt) if prompt is not None else None,
                "completion_tokens": int(completion) if completion is not None else None,
                "total_tokens": int(total) if total is not None else None,
                "cached_prompt_tokens": int(cached) if cached is not None else None,
                "cache_hit_rate": hit,
                "response_text": debug["response_text"],
                "response_tool_calls": debug["response_tool_calls"],
                "response_phase": debug["response_phase"],
                "response_finish_reason": debug["response_finish_reason"],
            },
        )
    except Exception:
        logger.debug("[%s] failed to emit llm_usage event", agent, exc_info=True)


def _emit_llm_runtime_status(
    logger: logging.Logger,
    agent: str,
    message: str,
    *,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    is_final: bool = False,
) -> None:
    try:
        from ..display import DisplayManager

        DisplayManager()._emit(
            "status",
            {
                "message": message,
                "agent": agent,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "final": is_final,
                "kind": "llm_runtime",
            },
        )
    except Exception:
        logger.debug("[%s] failed to emit llm runtime status", agent, exc_info=True)


def _is_rate_limit_error(err: Exception) -> bool:
    status = getattr(err, "status", None)
    status_code = getattr(err, "status_code", None)
    if status == 429 or status_code == 429:
        return True

    message = str(err).lower()
    markers = [
        "429",
        "too many requests",
        "rate limit",
        "ratelimit",
        "ratelimitexceeded",
        "resource exhausted",
        "resource_exhausted",
        "quota will reset after",
        "exhausted your capacity",
    ]
    return any(marker in message for marker in markers)


def _parse_retry_after_seconds(value: Any) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None

    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(raw)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, retry_at.timestamp() - time.time())
    except Exception:
        return None


def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
    header_sets = []
    response = getattr(err, "response", None)
    if response is not None:
        header_sets.append(getattr(response, "headers", None))
    header_sets.append(getattr(err, "headers", None))

    for headers in header_sets:
        if not headers or not hasattr(headers, "get"):
            continue
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        parsed = _parse_retry_after_seconds(retry_after)
        if parsed is not None:
            return parsed
    return None


def _apply_retry_jitter(sleep_s: float, jitter_s: float) -> float:
    if jitter_s <= 0:
        return max(0.0, sleep_s)
    return max(0.0, sleep_s + random.uniform(0.0, jitter_s))


def _message_has_tool_calls(message: Any) -> bool:
    tool_calls = getattr(message, "tool_calls", None) or []
    return bool(tool_calls)


def _response_needs_followup(message: Any) -> Optional[str]:
    if _message_has_tool_calls(message):
        return None

    stats = extract_usage_stats(message)
    debug = extract_response_debug(message)
    finish_reason = str(debug.get("response_finish_reason") or "").strip().lower()
    response_text = str(debug.get("response_text") or "").strip()
    error_message = str(debug.get("response_error_message") or "").strip()
    completion = stats.get("completion_tokens")

    if finish_reason == "error":
        return None

    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return (
            "Your previous response likely failed to arrive. Reproduce the answer from the beginning, "
            "but split it into smaller parts. Keep each part concise and stop before hitting the output limit."
        )

    try:
        if completion is not None and int(completion) == 0 and not response_text:
            return (
                "Your previous response was empty. Retry now. Do not return an empty message. "
                "Either produce the answer directly or, if needed, emit the next tool call."
            )
    except Exception:
        pass

    if not response_text and finish_reason in {"", "stop"}:
        return (
            "Your previous response was blank. Retry now. Do not return an empty message. "
            "Either answer directly or emit the next tool call."
        )

    return None


def _raise_on_provider_error(message: Any) -> None:
    debug = extract_response_debug(message)
    finish_reason = str(debug.get("response_finish_reason") or "").strip().lower()
    error_message = str(debug.get("response_error_message") or "").strip()
    if finish_reason != "error":
        return
    raise RuntimeError(error_message or "LLM provider returned finish_reason=error")


def derive_bounded_stateful_session_id(
    base_session: str,
    suffix: str,
    *,
    max_len: int = STATEFUL_SESSION_ID_MAX_LEN,
) -> str:
    """Derive a stable stateful session id that stays within provider limits."""
    base = str(base_session or "").strip() or "cytobridge"
    suffix_text = str(suffix or "").strip()
    if not suffix_text:
        return base[:max_len]

    candidate = f"{base}{suffix_text}"
    if len(candidate) <= max_len:
        return candidate

    keep = max_len - len(suffix_text)
    if keep <= 0:
        return suffix_text[-max_len:]
    return f"{base[:keep]}{suffix_text}"


def _append_followup_prompt(messages: Any, response: Any, followup_prompt: str) -> Any:
    if not isinstance(messages, list):
        return messages
    return list(messages) + [response, HumanMessage(content=followup_prompt)]


def _rotate_stateful_llm_session_if_needed(model: Any, response: Any, logger: logging.Logger, agent: str) -> None:
    """Reset stateful provider session IDs after blank replies so semantic retries use a fresh session."""
    if model is None or not hasattr(model, "session_id"):
        return

    stats = extract_usage_stats(response)
    debug = extract_response_debug(response)
    response_text = str(debug.get("response_text") or "").strip()
    finish_reason = str(debug.get("response_finish_reason") or "").strip().lower()
    completion = stats.get("completion_tokens")

    is_blank = False
    try:
        is_blank = completion is not None and int(completion) == 0 and not response_text
    except Exception:
        is_blank = False
    if not is_blank and not (not response_text and finish_reason in {"", "stop"}):
        return

    old_session = str(getattr(model, "session_id", "") or "").strip()
    if not old_session:
        return

    new_session = derive_bounded_stateful_session_id(
        old_session,
        f"-retry-{uuid.uuid4().hex[:8]}",
    )
    try:
        setattr(model, "session_id", new_session)
    except Exception:
        return

    logger.warning(
        "[%s] Rotated stateful LLM session after blank response: %s -> %s",
        agent,
        old_session,
        new_session,
    )


def invoke_with_retry(
    model: Any,
    messages: Any,
    logger: logging.Logger,
    agent: str,
) -> Any:
    """Invoke model with retry policy. By default retries all provider errors."""
    runtime = get_llm_runtime_settings()
    max_attempts = max(1, int(runtime["max_retries"]))
    semantic_retry_max = max(0, int(runtime["semantic_retry_max"]))
    backoff = float(runtime["retry_backoff_sec"])
    jitter = float(runtime["retry_jitter_sec"])
    rate_limit_sleep = float(runtime["rate_limit_retry_sec"])
    retry_all = bool(runtime["retry_all_errors"])

    last_err: Optional[Exception] = None
    semantic_retries = 0
    current_messages = messages
    for attempt in range(1, max_attempts + 1):
        try:
            response = model.invoke(current_messages)
            log_llm_response_stats(logger, agent, response)
            _raise_on_provider_error(response)
            followup_prompt = _response_needs_followup(response)
            if (
                followup_prompt
                and isinstance(current_messages, list)
                and semantic_retries < semantic_retry_max
                and attempt < max_attempts
            ):
                _rotate_stateful_llm_session_if_needed(model, response, logger, agent)
                semantic_retries += 1
                retry_msg = (
                    f"[{agent}] LLM response was truncated or empty; requesting continuation "
                    f"(semantic retry {semantic_retries}/{semantic_retry_max}, attempt {attempt}/{max_attempts})"
                )
                logger.warning("%s", retry_msg)
                _emit_llm_runtime_status(
                    logger,
                    agent,
                    retry_msg,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                current_messages = _append_followup_prompt(current_messages, response, followup_prompt)
                continue
            return response
        except Exception as e:  # noqa: BLE001
            last_err = e
            error_text = str(e).strip() or e.__class__.__name__

            # If disabled globally, retry only common transient markers.
            should_retry = attempt < max_attempts
            if should_retry and not retry_all:
                msg = str(e).lower()
                transient_markers = [
                    "timeout",
                    "timed out",
                    "connection reset",
                    "temporarily unavailable",
                    "429",
                    "rate limit",
                    "bad gateway",
                    "service unavailable",
                    "gateway timeout",
                ]
                if not any(marker in msg for marker in transient_markers):
                    should_retry = False

            if not should_retry:
                final_msg = f"[{agent}] LLM invoke failed after {attempt}/{max_attempts} attempts: {error_text}"
                logger.error(final_msg)
                _emit_llm_runtime_status(
                    logger,
                    agent,
                    final_msg,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    is_final=True,
                )
                break

            if _is_rate_limit_error(e):
                retry_after_s = _extract_retry_after_seconds(e)
                sleep_s = _apply_retry_jitter(
                    retry_after_s if retry_after_s is not None else rate_limit_sleep,
                    jitter,
                )
                retry_source = "Retry-After" if retry_after_s is not None else "default rate-limit backoff"
                retry_msg = (
                    f"[{agent}] LLM invoke hit rate limit (attempt {attempt}/{max_attempts}): "
                    f"{error_text}. Retrying in {sleep_s:.1f}s ({retry_source})"
                )
            else:
                sleep_s = _apply_retry_jitter(backoff * (2 ** (attempt - 1)), jitter)
                retry_msg = (
                    f"[{agent}] LLM invoke failed (attempt {attempt}/{max_attempts}): "
                    f"{error_text}. Retrying in {sleep_s:.1f}s"
                )
            logger.warning(
                "%s",
                retry_msg,
            )
            _emit_llm_runtime_status(
                logger,
                agent,
                retry_msg,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            time.sleep(sleep_s)

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"[{agent}] invoke_with_retry failed without error")
