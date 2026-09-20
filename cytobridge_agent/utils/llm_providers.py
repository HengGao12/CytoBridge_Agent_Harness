from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class LLMProvider:
    id: str
    display_name: str
    auth_mode: str = "api_key"
    transport: str = "openai_chat"
    base_url: str = ""
    base_url_env_var: str = ""
    api_key_env_vars: Tuple[str, ...] = ("OPENAI_API_KEY",)
    reasoning_mode: str = "none"
    model_ids: Tuple[str, ...] = ()
    default_model: str = ""
    context_window: int = 200000
    model_context_windows: Tuple[Tuple[str, int], ...] = ()
    supported: bool = True


LLM_PROVIDER_ALIASES: Dict[str, str] = {
    "": "auto",
    "default": "auto",
    "api_key": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "custom": "openai-compatible",
    "openai": "openai",
    "openai-api": "openai",
    "codex": "openai-codex",
    "codex_oauth": "openai-codex",
    "openai_codex": "openai-codex",
    "openai-codex-oauth": "openai-codex",
    "gemini": "google-gemini-cli",
    "gemini_oauth": "google-gemini-cli",
    "google_oauth": "google-gemini-cli",
    "google-gemini": "google-gemini-cli",
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding",
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "nvidia-nim": "nvidia",
    "nim": "nvidia",
    "dashscope": "alibaba",
    "qwen": "alibaba",
    "ai-gateway": "vercel",
    "vercel-ai-gateway": "vercel",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
}

MODELS_DEV_PROVIDER_ALIASES: Dict[str, Tuple[str, ...]] = {
    "google-gemini-cli": ("google",),
    "kimi-for-coding": ("kimi-for-coding", "moonshotai"),
}


LLM_PROVIDERS: Dict[str, LLMProvider] = {
    "auto": LLMProvider(
        id="auto",
        display_name="Auto",
        auth_mode="auto",
        transport="auto",
        api_key_env_vars=("OPENAI_API_KEY",),
        model_ids=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-4o"),
        context_window=200000,
    ),
    "openai-compatible": LLMProvider(
        id="openai-compatible",
        display_name="OpenAI-compatible",
        auth_mode="api_key",
        api_key_env_vars=("OPENAI_API_KEY",),
        model_ids=("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-4o"),
        context_window=200000,
    ),
    "openai": LLMProvider(
        id="openai",
        display_name="OpenAI API",
        auth_mode="api_key",
        base_url="https://api.openai.com/v1",
        api_key_env_vars=("OPENAI_API_KEY",),
        reasoning_mode="openai_reasoning_effort",
        model_ids=(
            "gpt-5.5",
            "gpt-5.5-pro",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
            "o3",
            "o3-mini",
        ),
        default_model="gpt-5.5",
    ),
    "openrouter": LLMProvider(
        id="openrouter",
        display_name="OpenRouter",
        auth_mode="api_key",
        base_url="https://openrouter.ai/api/v1",
        base_url_env_var="OPENROUTER_BASE_URL",
        api_key_env_vars=("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        reasoning_mode="openrouter_extra_body",
        model_ids=(
            "openai/gpt-5.5",
            "openai/gpt-5.4",
            "openai/gpt-5.4-mini",
            "openai/gpt-5.3-codex",
            "anthropic/claude-opus-4.7",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-sonnet-4.5",
            "moonshotai/kimi-k2.6",
            "xiaomi/mimo-v2.5-pro",
            "z-ai/glm-5.1",
            "x-ai/grok-4.20",
            "google/gemini-3.1-pro-preview",
            "qwen/qwen3.6-plus",
            "minimax/minimax-m2.7",
        ),
        default_model="openai/gpt-5.5",
        context_window=200000,
    ),
    "xiaomi": LLMProvider(
        id="xiaomi",
        display_name="Xiaomi MiMo",
        auth_mode="api_key",
        base_url="https://api.xiaomimimo.com/v1",
        base_url_env_var="XIAOMI_BASE_URL",
        api_key_env_vars=("XIAOMI_API_KEY",),
        reasoning_mode="deepseek_thinking",
        model_ids=("mimo-v2.5", "mimo-v2.5-pro", "mimo-v2-pro", "mimo-v2-omni", "mimo-v2-flash"),
        default_model="mimo-v2.5",
    ),
    "deepseek": LLMProvider(
        id="deepseek",
        display_name="DeepSeek",
        auth_mode="api_key",
        base_url="https://api.deepseek.com/v1",
        base_url_env_var="DEEPSEEK_BASE_URL",
        api_key_env_vars=("DEEPSEEK_API_KEY",),
        reasoning_mode="deepseek_thinking",
        model_ids=("deepseek-v4-pro", "deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"),
        default_model="deepseek-v4-flash",
        context_window=1000000,
    ),
    "zai": LLMProvider(
        id="zai",
        display_name="Z.AI / GLM",
        auth_mode="api_key",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        base_url_env_var="GLM_BASE_URL",
        api_key_env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        model_ids=("glm-5.1", "glm-5", "glm-5v-turbo", "glm-5-turbo", "glm-4.7", "glm-4.5"),
        default_model="glm-5.1",
    ),
    "kimi-for-coding": LLMProvider(
        id="kimi-for-coding",
        display_name="Kimi / Moonshot",
        auth_mode="api_key",
        base_url="https://api.kimi.com/v1",
        base_url_env_var="KIMI_BASE_URL",
        api_key_env_vars=("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
        reasoning_mode="kimi_thinking",
        model_ids=("kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2-turbo-preview"),
        default_model="kimi-k2.6",
    ),
    "alibaba": LLMProvider(
        id="alibaba",
        display_name="Alibaba / DashScope",
        auth_mode="api_key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        base_url_env_var="DASHSCOPE_BASE_URL",
        api_key_env_vars=("DASHSCOPE_API_KEY",),
        model_ids=("qwen3.6-plus", "qwen3.5-plus", "qwen3-coder-plus", "qwen3-coder-next", "glm-5", "kimi-k2.5"),
        default_model="qwen3.6-plus",
    ),
    "nvidia": LLMProvider(
        id="nvidia",
        display_name="NVIDIA NIM",
        auth_mode="api_key",
        base_url="https://integrate.api.nvidia.com/v1",
        base_url_env_var="NVIDIA_BASE_URL",
        api_key_env_vars=("NVIDIA_API_KEY",),
        model_ids=(
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/nemotron-3-nano-30b-a3b",
            "deepseek-ai/deepseek-v3.2",
            "moonshotai/kimi-k2.6",
            "openai/gpt-oss-120b",
        ),
        default_model="nvidia/nemotron-3-super-120b-a12b",
    ),
    "xai": LLMProvider(
        id="xai",
        display_name="xAI",
        auth_mode="api_key",
        base_url="https://api.x.ai/v1",
        base_url_env_var="XAI_BASE_URL",
        api_key_env_vars=("XAI_API_KEY",),
        model_ids=("grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning", "grok-4", "grok-code-fast-1"),
        default_model="grok-4",
    ),
    "vercel": LLMProvider(
        id="vercel",
        display_name="Vercel AI Gateway",
        auth_mode="api_key",
        base_url="https://ai-gateway.vercel.sh/v1",
        base_url_env_var="AI_GATEWAY_BASE_URL",
        api_key_env_vars=("AI_GATEWAY_API_KEY",),
        reasoning_mode="openrouter_extra_body",
        model_ids=(
            "openai/gpt-5.4",
            "openai/gpt-5.4-mini",
            "anthropic/claude-sonnet-4.6",
            "moonshotai/kimi-k2.6",
            "zai/glm-5.1",
            "xai/grok-4.20-reasoning",
        ),
        default_model="openai/gpt-5.4",
        context_window=200000,
    ),
    "google-gemini-cli": LLMProvider(
        id="google-gemini-cli",
        display_name="Google Gemini OAuth",
        auth_mode="gemini_oauth",
        transport="gemini_oauth",
        api_key_env_vars=(),
        reasoning_mode="gemini_oauth",
        model_ids=("gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro"),
        default_model="gemini-3.1-pro-preview",
    ),
    "openai-codex": LLMProvider(
        id="openai-codex",
        display_name="OpenAI Codex OAuth",
        auth_mode="codex_oauth",
        transport="codex_oauth",
        reasoning_mode="codex_oauth",
        api_key_env_vars=(),
        model_ids=(
            "gpt-5.6-sol",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.3-codex",
            "gpt-5.2",
        ),
        default_model="gpt-5.6-sol",
    ),
    "minimax": LLMProvider(
        id="minimax",
        display_name="MiniMax",
        transport="anthropic_messages",
        base_url_env_var="MINIMAX_BASE_URL",
        api_key_env_vars=("MINIMAX_API_KEY",),
        model_ids=("MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"),
        supported=False,
    ),
    "minimax-cn": LLMProvider(
        id="minimax-cn",
        display_name="MiniMax (China)",
        transport="anthropic_messages",
        base_url_env_var="MINIMAX_CN_BASE_URL",
        api_key_env_vars=("MINIMAX_CN_API_KEY",),
        model_ids=("MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2"),
        supported=False,
    ),
}


def normalize_llm_provider(value: Optional[str]) -> str:
    normalized = str(value or "auto").strip().lower().replace("_", "-")
    provider_id = LLM_PROVIDER_ALIASES.get(normalized, normalized)
    if provider_id in LLM_PROVIDERS:
        return provider_id
    return normalized


def get_llm_provider(provider: Optional[str]) -> LLMProvider:
    provider_id = normalize_llm_provider(provider)
    return LLM_PROVIDERS.get(
        provider_id,
        LLMProvider(
            id=provider_id,
            display_name=provider_id,
            auth_mode="api_key",
            api_key_env_vars=("OPENAI_API_KEY",),
        ),
    )


def _normalize_provider_base_url(spec: LLMProvider, value: str) -> str:
    normalized = str(value or "").strip()
    if spec.id == "deepseek" and normalized.rstrip("/") == "https://api.deepseek.com":
        return "https://api.deepseek.com/v1"
    return normalized


def resolve_provider_base_url(spec: LLMProvider, explicit_base_url: Optional[str]) -> Optional[str]:
    value = _normalize_provider_base_url(spec, str(explicit_base_url or "").strip())
    if value:
        return value
    if spec.base_url_env_var:
        env_value = _normalize_provider_base_url(spec, os.getenv(spec.base_url_env_var, "").strip())
        if env_value:
            return env_value
    return _normalize_provider_base_url(spec, spec.base_url or "") or None


def resolve_provider_api_key(spec: LLMProvider, explicit_api_key: Optional[str]) -> Optional[str]:
    value = str(explicit_api_key or "").strip()
    if value:
        return value
    for env_var in spec.api_key_env_vars:
        env_value = os.getenv(env_var, "").strip()
        if env_value:
            return env_value
    return None


def provider_model_ids(provider: Optional[str]) -> Tuple[str, ...]:
    return get_llm_provider(provider).model_ids


def normalize_model_for_provider(provider: Optional[str], model: Optional[str]) -> str:
    model_name = str(model or "").strip()
    spec = get_llm_provider(provider)
    if not model_name:
        return spec.default_model or "gpt-4o"
    lower_provider = spec.id
    prefix_map = {
        "openai": ("openai/",),
        "openai-codex": ("openai/", "openai-codex/"),
        "xiaomi": ("xiaomi/", "xiaomimimo/"),
        "zai": ("z-ai/", "zai/", "glm/"),
        "deepseek": ("deepseek/", "deepseek-ai/"),
        "kimi-for-coding": ("moonshotai/", "moonshot/", "kimi/"),
        "alibaba": ("qwen/", "alibaba/"),
        "xai": ("x-ai/", "xai/"),
    }
    if lower_provider in {"openrouter", "vercel", "nvidia", "openai-compatible", "auto"}:
        return model_name
    for prefix in prefix_map.get(lower_provider, ()):
        if model_name.lower().startswith(prefix):
            return model_name.split("/", 1)[1]
    return model_name


def _models_dev_cache_path() -> Path:
    explicit = os.getenv("CYTOBRIDGE_MODELS_DEV_CACHE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path(__file__).with_name("model_context_windows.json")


@lru_cache(maxsize=1)
def _load_models_dev_cache() -> Dict[str, Any]:
    path = _models_dev_cache_path()
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalize_model_id_for_context(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _context_from_flat_registry(
    registry: Dict[str, Any],
    provider_keys: Tuple[str, ...],
    candidate_ids: set[str],
) -> Optional[int]:
    providers = registry.get("providers")
    if not isinstance(providers, dict):
        return None
    for provider_key in provider_keys:
        models = providers.get(provider_key)
        if not isinstance(models, dict):
            continue
        for model_id, context in models.items():
            normalized_id = _normalize_model_id_for_context(model_id)
            model_id_tail = normalized_id.split("/")[-1] if "/" in normalized_id else normalized_id
            if normalized_id not in candidate_ids and model_id_tail not in candidate_ids:
                continue
            try:
                context_int = int(context)
            except Exception:
                continue
            if context_int > 0:
                return context_int
    return None


def _context_from_models_dev_cache(provider: Optional[str], model: Optional[str]) -> Optional[int]:
    cache = _load_models_dev_cache()
    if not cache:
        return None
    spec = get_llm_provider(provider)
    provider_keys = (spec.id, *MODELS_DEV_PROVIDER_ALIASES.get(spec.id, ()))
    normalized_model = normalize_model_for_provider(spec.id, model)
    candidate_ids = {
        _normalize_model_id_for_context(model),
        _normalize_model_id_for_context(normalized_model),
    }
    for item in list(candidate_ids):
        if "/" in item:
            candidate_ids.add(item.split("/")[-1])

    flat_context = _context_from_flat_registry(cache, provider_keys, candidate_ids)
    if flat_context is not None:
        return flat_context

    for provider_key in provider_keys:
        entry = cache.get(provider_key)
        if not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model_meta in models.items():
            normalized_id = _normalize_model_id_for_context(model_id)
            model_id_tail = normalized_id.split("/")[-1] if "/" in normalized_id else normalized_id
            if normalized_id not in candidate_ids and model_id_tail not in candidate_ids:
                continue
            if not isinstance(model_meta, dict):
                continue
            limit = model_meta.get("limit")
            if not isinstance(limit, dict):
                continue
            context = limit.get("context")
            try:
                context_int = int(context)
            except Exception:
                continue
            if context_int > 0:
                return context_int
    return None


def resolve_provider_context_window(
    provider: Optional[str],
    model: Optional[str],
    *,
    default: int = 200000,
) -> int:
    cached_context = _context_from_models_dev_cache(provider, model)
    if cached_context is not None:
        return int(cached_context)
    spec = get_llm_provider(provider)
    model_name = normalize_model_for_provider(spec.id, model).strip().lower()
    for candidate, window in spec.model_context_windows:
        if model_name == str(candidate).strip().lower():
            return int(window)
    try:
        return int(spec.context_window or default)
    except Exception:
        return int(default)


def _clamp_effort(effort: str, allowed: Tuple[str, ...], default: str) -> str:
    value = str(effort or "").strip().lower()
    if value in {"off", "none"}:
        return "off"
    if value in {"minimal", "min"}:
        value = "minimal"
    if value in {"medium", "med", "mid"}:
        value = "medium"
    if value in {"xhigh", "extra_high", "extra-high", "max"}:
        value = "xhigh"
    if value in allowed:
        return value
    return default


def build_reasoning_kwargs(
    provider: Optional[str],
    model: str,
    effort: str,
    *,
    openai_reasoning_supported: bool = False,
    force_openai_reasoning: bool = False,
) -> Dict[str, Any]:
    spec = get_llm_provider(provider)
    normalized_effort = str(effort or "").strip().lower()
    if not normalized_effort:
        return {}
    if spec.reasoning_mode == "openrouter_extra_body":
        if normalized_effort == "off":
            return {"extra_body": {"reasoning": {"enabled": False}}}
        clamped = _clamp_effort(normalized_effort, ("minimal", "low", "medium", "high", "xhigh"), "medium")
        return {"extra_body": {"reasoning": {"enabled": True, "effort": clamped}}}
    if spec.reasoning_mode == "kimi_thinking":
        if normalized_effort == "off":
            return {
                "reasoning_effort": "low",
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        clamped = _clamp_effort(normalized_effort, ("low", "medium", "high"), "medium")
        return {
            "reasoning_effort": clamped,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    if spec.reasoning_mode == "deepseek_thinking":
        # DeepSeek V4 uses a provider-specific thinking switch instead of
        # top-level OpenAI-style reasoning_effort. Its effort is nested under
        # thinking and currently supports only high/max.
        if normalized_effort == "off":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        clamped = _clamp_effort(normalized_effort, ("high", "xhigh"), "high")
        deepseek_effort = "max" if clamped == "xhigh" else "high"
        return {"extra_body": {"thinking": {"type": "enabled", "reasoning_effort": deepseek_effort}}}
    if spec.reasoning_mode == "openai_reasoning_effort" or spec.id in {"openai-compatible", "auto"}:
        if not openai_reasoning_supported and not force_openai_reasoning:
            return {}
        if normalized_effort == "off":
            return {"reasoning_effort": "none"}
        clamped = _clamp_effort(normalized_effort, ("minimal", "low", "medium", "high", "xhigh"), "medium")
        return {"reasoning_effort": clamped}
    return {}


def provider_options() -> list[dict[str, Any]]:
    return [
        {
            "id": spec.id,
            "display_name": spec.display_name,
            "transport": spec.transport,
            "auth_mode": spec.auth_mode,
            "reasoning_mode": spec.reasoning_mode,
            "default_model": spec.default_model,
            "context_window": spec.context_window,
            "model_context_windows": dict(spec.model_context_windows),
            "model_ids": list(spec.model_ids),
            "supported": "true" if spec.supported else "false",
        }
        for spec in LLM_PROVIDERS.values()
    ]
