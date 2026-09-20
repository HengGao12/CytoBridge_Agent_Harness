"""Agent selection helpers for benchmark runners.

The global CytoBridge config may contain a specific Codex OAuth profile.  Benchmark
runs should still be able to select another agent explicitly without editing that
global file, so this module centralizes the override rules.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


AGENT_TYPE_ALIASES = {
    "cytobridge": "cytobridge",
    "codex": "codex",
    "openai-codex": "codex",
    "openai_codex": "codex",
    "biomni": "biomini",
    "biomini": "biomini",
    "bio-mini": "biomini",
}


@dataclass(frozen=True)
class AgentSelection:
    agent_type: str
    runner_id: str
    run_label: str
    runner_kwargs: dict[str, Any]


def normalize_agent_type(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = (
            os.environ.get("CYTOBRIDGE_AGENT_TYPE")
            or os.environ.get("BENCHMARK_AGENT_TYPE")
            or "codex"
        ).strip().lower()
    normalized = AGENT_TYPE_ALIASES.get(raw)
    if not normalized:
        valid = ", ".join(sorted(set(AGENT_TYPE_ALIASES.values())))
        raise ValueError(f"Unknown agent_type '{value}'. Expected one of: {valid}.")
    return normalized


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_saved_config() -> dict[str, Any]:
    """Read ~/.cellcompass/config.json without importing the full agent package."""
    config_file = Path.home() / ".cellcompass" / "config.json"
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _saved_api_key(saved_config: dict[str, Any]) -> Optional[str]:
    """Resolve API key from current and legacy CytoBridge config shapes."""
    explicit = saved_config.get("openai_api_key")
    if explicit:
        return str(explicit)
    provider_keys = saved_config.get("provider_api_keys")
    if not isinstance(provider_keys, dict) or not provider_keys:
        return None
    provider = str(saved_config.get("llm_provider") or "").strip()
    if provider and provider_keys.get(provider):
        return str(provider_keys[provider])
    if len(provider_keys) == 1:
        return str(next(iter(provider_keys.values())))
    base_url = str(saved_config.get("llm_base_url") or "").lower()
    if "xiaomimimo" in base_url and provider_keys.get("xiaomi"):
        return str(provider_keys["xiaomi"])
    return None


def _saved_base_url(saved_config: dict[str, Any]) -> Optional[str]:
    """Resolve provider-specific base URL from current and legacy config shapes."""
    provider = str(saved_config.get("llm_provider") or "").strip()
    provider_base_urls = saved_config.get("provider_base_urls")
    if provider and isinstance(provider_base_urls, dict) and provider_base_urls.get(provider):
        return str(provider_base_urls[provider])
    explicit = saved_config.get("llm_base_url")
    if explicit:
        return str(explicit)
    return None


def get_current_codex_profile_id(saved_config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return the configured/current usable Codex OAuth profile id."""
    saved_config = saved_config or {}
    configured = str(saved_config.get("llm_profile_id") or "").strip()
    if configured:
        return configured

    auth_file = Path.home() / ".cellcompass" / "auth_profiles.json"
    if not auth_file.exists():
        return None
    try:
        auth = json.loads(auth_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    last_good = str((auth.get("last_good") or {}).get("openai-codex") or "").strip()
    if last_good:
        return last_good
    profiles = auth.get("profiles") or {}
    for profile_id in (auth.get("order") or {}).get("openai-codex", []):
        profile = profiles.get(profile_id) or {}
        if profile.get("enabled", True) and not profile.get("reauth_required", False):
            return str(profile_id)
    return None


def resolve_agent_selection(
    *,
    agent_type: Optional[str],
    saved_config: dict[str, Any],
    llm_model: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_auth_mode: Optional[str] = None,
    llm_profile_id: Optional[str] = None,
    llm_thinking_level: Optional[str] = None,
    harness_revisions: Optional[int] = None,
    biomni_path: Optional[str] = None,
    biomni_source: Optional[str] = None,
) -> AgentSelection:
    """Resolve CLI/env/config into a concrete benchmark runner selection."""
    selected = normalize_agent_type(agent_type)

    if selected == "cytobridge":
        provider = llm_provider or _env_first("CYTOBRIDGE_LLM_PROVIDER") or saved_config.get("llm_provider", "auto")
        profile_id = llm_profile_id or _env_first("CYTOBRIDGE_LLM_PROFILE_ID") or saved_config.get("llm_profile_id") or get_current_codex_profile_id(saved_config)
        return AgentSelection(
            agent_type=selected,
            runner_id="cytobridge",
            run_label="cytobridge",
            runner_kwargs={
                "llm_model": llm_model or saved_config.get("llm_model"),
                "llm_provider": provider,
                "llm_base_url": llm_base_url or _saved_base_url(saved_config),
                "llm_api_key": llm_api_key or _env_first("OPENAI_API_KEY", "CYTOBRIDGE_OPENAI_API_KEY") or _saved_api_key(saved_config),
                "llm_auth_mode": llm_auth_mode or saved_config.get("llm_auth_mode", "openai-codex"),
                "llm_profile_id": profile_id,
                "llm_thinking_level": llm_thinking_level or saved_config.get("llm_thinking_level", "low"),
                "harness_revisions": 1 if harness_revisions is None else max(0, int(harness_revisions)),
            },
        )

    if selected == "codex":
        # Codex CLI profiles are config.toml profiles, not CytoBridge OAuth profile
        # ids.  Do not pass the CytoBridge auth profile through as `codex --profile`.
        profile_id = llm_profile_id or _env_first("CODEX_PROFILE")
        return AgentSelection(
            agent_type=selected,
            runner_id="codex",
            run_label="codex",
            runner_kwargs={
                "llm_model": llm_model or _env_first("CYTOBRIDGE_LLM_MODEL") or "gpt-5.5",
                "llm_profile_id": profile_id,
            },
        )

    profile_id = llm_profile_id or _env_first("CYTOBRIDGE_LLM_PROFILE_ID") or get_current_codex_profile_id(saved_config)
    return AgentSelection(
        agent_type=selected,
        runner_id="biomini",
        run_label="biomini",
        runner_kwargs={
            "biomni_repo": biomni_path or _env_first("BIOMNI_REPO", "BIOMNI_PATH") or str(Path(__file__).resolve().parent.parent / "Biomni"),
            "llm_model": llm_model or _env_first("CYTOBRIDGE_LLM_MODEL") or "gpt-5.4",
            "llm_source": biomni_source or _env_first("BIOMNI_SOURCE"),
            "llm_base_url": llm_base_url or _env_first("BIOMNI_CUSTOM_BASE_URL"),
            "llm_api_key": llm_api_key or _env_first("BIOMNI_CUSTOM_API_KEY"),
            "llm_auth_mode": "codex_oauth",
            "llm_profile_id": profile_id,
            "llm_thinking_level": llm_thinking_level or saved_config.get("llm_thinking_level", "off"),
        },
    )
