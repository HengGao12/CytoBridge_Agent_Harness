from __future__ import annotations

import os
from typing import Optional, Tuple

from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .llm_runtime import get_llm_runtime_settings
from .llm_providers import (
    build_reasoning_kwargs,
    get_llm_provider,
    normalize_llm_provider,
    normalize_model_for_provider,
    resolve_provider_api_key,
    resolve_provider_base_url,
)


CODEX_ALLOWED_MODEL_IDS = {
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-sol",
}

OPENAI_REASONING_EFFORT_MODEL_IDS = {
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
    *CODEX_ALLOWED_MODEL_IDS,
}

CODEX_ALLOWED_THINKING_LEVELS = {
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}


def normalize_auth_mode(value: Optional[str]) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"gemini_oauth", "google_oauth", "oauth"}:
        return "gemini_oauth"
    if mode in {"codex_oauth", "openai_codex_oauth", "codex"}:
        return "codex_oauth"
    if mode in {"api_key", "openai_compatible"}:
        return "api_key"
    return "auto"


def normalize_base_url_for_model(model: str, base_url: Optional[str]) -> Optional[str]:
    value = (base_url or "").strip()
    if not model.startswith("gemini"):
        return value or None
    if value in {"", "https://api.openai.com/v1", "https://api.openai.com/v1/"}:
        return None
    return value


def is_codex_model(model: str) -> bool:
    selected = str(model or "").strip()
    normalized = normalize_codex_model_name(selected).lower()
    return normalized in CODEX_ALLOWED_MODEL_IDS


def normalize_codex_model_name(model: str) -> str:
    selected = str(model or "").strip()
    if selected.startswith("openai-codex/"):
        return selected.split("/", 1)[1]
    if selected.startswith("openai/"):
        return selected.split("/", 1)[1]
    return selected


def normalize_llm_reasoning_effort(value: Optional[str], *, default: str = "low") -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    if raw in {"off", "none", "disable", "disabled", "false", "0"}:
        return "off"
    if raw in {"min", "minimal"}:
        return "minimal"
    if raw in {"low", "on", "enable", "enabled", "true", "1"}:
        return "low"
    if raw in {"med", "mid", "medium"}:
        return "medium"
    if raw in {"high"}:
        return "high"
    if raw in {"xhigh", "extra_high", "extra-high", "max"}:
        return "xhigh"
    raise ValueError(
        "Invalid reasoning effort. Expected one of: off, minimal, low, medium, high, xhigh."
    )


def normalize_codex_thinking_level(value: Optional[str], *, default: str = "low") -> str:
    return normalize_llm_reasoning_effort(value, default=default)


def _model_supports_openai_reasoning_effort(model: str) -> bool:
    normalized = normalize_codex_model_name(str(model or "").strip()).lower()
    if not normalized:
        return False
    return normalized in OPENAI_REASONING_EFFORT_MODEL_IDS


def _force_openai_reasoning_effort() -> bool:
    raw = os.getenv("CYTOBRIDGE_LLM_FORCE_REASONING_EFFORT", "").strip().lower()
    return raw in {"1", "true", "yes", "on", "enable", "enabled"}


def _openai_reasoning_kwargs(
    model: str,
    effort_value: Optional[str],
    *,
    provider: Optional[str] = None,
) -> dict:
    effort = normalize_llm_reasoning_effort(effort_value, default="").strip()
    if not effort:
        return {}
    return build_reasoning_kwargs(
        provider,
        model,
        effort,
        openai_reasoning_supported=_model_supports_openai_reasoning_effort(model),
        force_openai_reasoning=_force_openai_reasoning_effort(),
    )


def _configure_reasoning_content_roundtrip_patch(provider_id: str) -> None:
    """Preserve provider thinking traces for OpenAI-compatible round trips.

    Some thinking-mode APIs return assistant `reasoning_content` beside
    `content` and require it to be echoed back after tool calls. The current
    langchain-openai converter drops that provider-specific field, so the next
    request can fail with a provider-side 400 unless we round-trip it through
    AIMessage.additional_kwargs. Keep the echo path scoped to providers that
    have demonstrated this requirement.
    """

    try:
        import langchain_openai.chat_models.base as openai_base
    except Exception:
        return
    current_provider = str(provider_id or "").strip().lower()
    openai_base._cytobridge_reasoning_content_provider = current_provider
    if getattr(openai_base, "_cytobridge_reasoning_content_patch", False):
        return

    original_dict_to_message = openai_base._convert_dict_to_message
    original_message_to_dict = openai_base._convert_message_to_dict

    def _reasoning_roundtrip_provider() -> str:
        provider = str(getattr(openai_base, "_cytobridge_reasoning_content_provider", "") or "").strip().lower()
        return provider if provider in {"xiaomi", "deepseek"} else ""

    def _convert_dict_to_message_with_reasoning(raw: dict):
        message = original_dict_to_message(raw)
        try:
            if (
                _reasoning_roundtrip_provider()
                and raw.get("role") == "assistant"
                and "reasoning_content" in raw
                and isinstance(message, AIMessage)
            ):
                message.additional_kwargs["reasoning_content"] = raw.get("reasoning_content") or ""
                message.additional_kwargs["_cytobridge_reasoning_content_provider"] = _reasoning_roundtrip_provider()
        except Exception:
            pass
        return message

    def _convert_message_to_dict_with_reasoning(message, *args, **kwargs):
        payload = original_message_to_dict(message, *args, **kwargs)
        try:
            additional = getattr(message, "additional_kwargs", {}) or {}
            provider_marker = str(additional.get("_cytobridge_reasoning_content_provider") or "").strip().lower()
            has_reasoning_content = "reasoning_content" in additional
            reasoning_content = additional.get("reasoning_content")
            payload.pop("_cytobridge_reasoning_content_provider", None)
            current_provider = _reasoning_roundtrip_provider()
            if not (current_provider and provider_marker == current_provider):
                payload.pop("reasoning_content", None)
            if (
                current_provider
                and provider_marker == current_provider
                and isinstance(message, AIMessage)
                and has_reasoning_content
            ):
                payload["reasoning_content"] = reasoning_content
        except Exception:
            pass
        return payload

    openai_base._convert_dict_to_message = _convert_dict_to_message_with_reasoning
    openai_base._convert_message_to_dict = _convert_message_to_dict_with_reasoning
    openai_base._cytobridge_reasoning_content_patch = True


def instantiate_llm(
    *,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    auth_mode: Optional[str],
    provider: Optional[str] = None,
    preferred_profile_id: Optional[str] = None,
    thinking_level: Optional[str] = None,
) -> Tuple[BaseChatModel, str]:
    provider_id = normalize_llm_provider(provider)
    provider_spec = get_llm_provider(provider_id)
    model_name = normalize_model_for_provider(provider_id, model)
    normalized_auth_mode = normalize_auth_mode(auth_mode)
    if provider_id != "auto":
        normalized_auth_mode = provider_spec.auth_mode
        if not provider_spec.supported:
            raise ValueError(
                f"Provider '{provider_spec.id}' uses transport '{provider_spec.transport}', "
                "which is not supported by this CytoBridge runtime yet."
            )
    runtime = get_llm_runtime_settings()

    if normalized_auth_mode == "gemini_oauth":
        normalized_base_url = None
    elif provider_id != "auto":
        normalized_base_url = resolve_provider_base_url(provider_spec, base_url)
    elif normalized_auth_mode == "auto":
        normalized_base_url = normalize_base_url_for_model(model_name, base_url)
    else:
        normalized_base_url = (base_url or "").strip() or None

    is_gemini_oauth = (
        normalized_auth_mode == "gemini_oauth"
        or (normalized_auth_mode == "auto" and model_name.startswith("gemini") and not normalized_base_url)
    )
    is_codex_oauth = (
        normalized_auth_mode == "codex_oauth"
        or (normalized_auth_mode == "auto" and is_codex_model(model_name) and not normalized_base_url)
    )

    if normalized_auth_mode == "gemini_oauth" and not model_name.startswith("gemini"):
        raise ValueError("Gemini OAuth mode requires a Gemini model.")

    if is_gemini_oauth:
        from .cloud_code_chat import ChatCloudCodeAssist
        from .gemini_oauth import get_valid_access_token, get_valid_auth_context

        auth = get_valid_auth_context()
        llm = ChatCloudCodeAssist(
            model_name=model_name,
            temperature=0.0,
            api_key=auth["access_token"],
            project_id=auth.get("project_id") or "none",
            token_provider=get_valid_access_token,
            request_timeout_sec=float(runtime["timeout_sec"]),
            max_retries=1,
        )
        return llm, "gemini_oauth"

    if normalized_auth_mode == "codex_oauth" and not is_codex_model(model_name):
        raise ValueError("Codex OAuth mode requires a Codex model.")

    if is_codex_oauth:
        from .codex_chat import ChatCodexOAuth

        llm = ChatCodexOAuth(
            model_name=normalize_codex_model_name(model_name),
            temperature=0.0,
            max_tokens=10000,
            preferred_profile_id=(preferred_profile_id or "").strip() or None,
            thinking_level=normalize_codex_thinking_level(thinking_level, default="low"),
        )
        return llm, "codex_oauth"

    if normalized_auth_mode == "api_key" and model_name.startswith("gemini") and not normalized_base_url:
        raise ValueError("Gemini API-key mode requires an OpenAI-compatible llm_base_url.")

    if provider_id != "auto":
        resolved_api_key = resolve_provider_api_key(provider_spec, api_key)
    else:
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        env_hint = ""
        if provider_id != "auto" and provider_spec.api_key_env_vars:
            env_hint = f" Expected one of env vars: {', '.join(provider_spec.api_key_env_vars)}."
        raise ValueError(f"No API key available for provider '{provider_spec.id}'. Please configure settings first.{env_hint}")

    from langchain_openai import ChatOpenAI

    reasoning_kwargs = _openai_reasoning_kwargs(model_name, thinking_level, provider=provider_id)
    _configure_reasoning_content_roundtrip_patch(provider_id)
    llm = ChatOpenAI(
        model=model_name,
        max_tokens=10000,
        temperature=0,
        api_key=resolved_api_key,
        base_url=normalized_base_url,
        timeout=float(runtime["timeout_sec"]),
        max_retries=int(runtime["sdk_max_retries"]),
        **reasoning_kwargs,
    )
    return llm, provider_id if provider_id != "auto" else "openai_compatible"
