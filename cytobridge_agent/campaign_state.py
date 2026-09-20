"""Canonical campaign-state resolution helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def resolve_active_campaign(state: Dict[str, Any], *, file_tools: Optional[Any] = None) -> Dict[str, Any]:
    """Resolve the active campaign with registry-first semantics.

    Session state intentionally stores a compact campaign summary for prompt
    continuity, but the campaign JSON registry is authoritative for gates,
    locks, stage panels, and lifecycle status.
    """

    state = _as_dict(state)
    state_campaign = _as_dict(state.get("active_algorithm_campaign"))
    campaign_id = _text(state.get("active_algorithm_campaign_id") or state_campaign.get("campaign_id"))
    resolution: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "source": "none",
        "campaign": {},
        "registry_error": "",
        "state_has_campaign": bool(state_campaign),
        "used_state_fallback": False,
    }
    if campaign_id and file_tools is not None:
        try:
            campaign = _as_dict(file_tools.get_algorithm_campaign_status(campaign_id=campaign_id))
            if campaign:
                resolution.update(
                    {
                        "campaign": campaign,
                        "campaign_id": _text(campaign.get("campaign_id") or campaign_id),
                        "source": "registry",
                    }
                )
                return resolution
        except Exception as exc:
            resolution["registry_error"] = f"{exc.__class__.__name__}: {exc}"
    if state_campaign:
        resolution.update(
            {
                "campaign": state_campaign,
                "campaign_id": _text(state_campaign.get("campaign_id") or campaign_id),
                "source": "session_state",
                "used_state_fallback": True,
            }
        )
    return resolution


def state_with_resolved_campaign(state: Dict[str, Any], resolution: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow state copy with the resolved campaign bound."""

    merged = dict(state or {})
    campaign = _as_dict(resolution.get("campaign"))
    campaign_id = _text(resolution.get("campaign_id") or campaign.get("campaign_id"))
    if campaign:
        merged["active_algorithm_campaign"] = campaign
    if campaign_id:
        merged["active_algorithm_campaign_id"] = campaign_id
    return merged
