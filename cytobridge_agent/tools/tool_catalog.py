"""Lightweight tool catalog with deferred activation support."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

VALID_ACTIVATION_MODES = {"auto_next_turn", "manual_review"}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip()).strip("_")
    if not base:
        base = f"tool_{uuid4().hex[:8]}"
    if not re.match(r"^[A-Za-z_]", base):
        base = f"tool_{base}"
    return base.lower()


class ToolCatalog:
    """State-backed tool registry for builtin and generated tools."""

    def __init__(
        self,
        state: Dict[str, Any],
        output_dir: Path,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.state = state
        self.output_dir = Path(output_dir)
        self.event_sink = event_sink
        self.generated_dir = self.output_dir / "generated_tools"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        self.state.setdefault("tool_catalog", {})
        self.state.setdefault("pending_tools", [])
        self.state.setdefault("tool_catalog_revision", 0)
        self.state.setdefault("tool_activation_mode", "auto_next_turn")
        self.state.setdefault("tool_harvest_enabled", False)
        self.state.setdefault("tool_harvest_report", {})
        self.state.setdefault("tool_review_report", {})

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            logger.debug("ToolCatalog event sink failed", exc_info=True)

    def get_activation_mode(self) -> str:
        mode = self.state.get("tool_activation_mode", "auto_next_turn")
        if mode not in VALID_ACTIVATION_MODES:
            return "auto_next_turn"
        return mode

    def set_activation_mode(self, mode: str) -> str:
        if mode not in VALID_ACTIVATION_MODES:
            raise ValueError(f"Invalid activation mode: {mode}")
        self.state["tool_activation_mode"] = mode
        return mode

    def register_builtin(self, name: str, description: str, method_name: str) -> Dict[str, Any]:
        catalog = self.state["tool_catalog"]
        if name in catalog:
            existing = catalog[name]
            if existing.get("kind") == "builtin":
                return existing

        spec = {
            "tool_id": f"builtin::{name}",
            "name": name,
            "description": description,
            "kind": "builtin",
            "status": "active",
            "version": 1,
            "method_name": method_name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "requires_human_approval": True,
        }
        catalog[name] = spec
        return spec

    def list_specs(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        items = list(self.state.get("tool_catalog", {}).values())
        if include_disabled:
            return items
        return [item for item in items if item.get("status") == "active"]

    def list_active_specs(self) -> List[Dict[str, Any]]:
        return self.list_specs(include_disabled=False)

    def list_pending(self) -> List[Dict[str, Any]]:
        return list(self.state.get("pending_tools", []))

    def _next_generated_name(self, requested_name: str) -> str:
        base = _slugify(requested_name)
        if not base.startswith("generated_"):
            base = f"generated_{base}"
        catalog = self.state.get("tool_catalog", {})
        if base not in catalog:
            return base

        suffix = 2
        while f"{base}_{suffix}" in catalog:
            suffix += 1
        return f"{base}_{suffix}"

    def queue_candidate(self, candidate: Dict[str, Any], current_turn: int) -> Dict[str, Any]:
        pending = self.state["pending_tools"]
        requested_name = candidate.get("name") or "generated_tool"
        name = self._next_generated_name(requested_name)
        tool_id = f"generated::{name}::{uuid4().hex[:8]}"

        queued = {
            "tool_id": tool_id,
            "name": name,
            "description": candidate.get("description", "Generated reusable tool"),
            "kind": "generated",
            "status": "pending_activation",
            "version": 1,
            "function_name": candidate.get("function_name") or "run_tool",
            "args_schema": candidate.get("args_schema") or {"type": "object", "properties": {}},
            "code": candidate.get("code", ""),
            "usage": candidate.get("usage", ""),
            "rationale": candidate.get("rationale", ""),
            "minimal_tests": candidate.get("minimal_tests", []),
            "created_turn": int(current_turn),
            "eligible_turn": int(current_turn) + 1,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "approved": self.get_activation_mode() == "auto_next_turn",
        }

        pending.append(queued)
        self._emit(
            "tool_activation_pending",
            {
                "tool_id": queued["tool_id"],
                "name": queued["name"],
                "eligible_turn": queued["eligible_turn"],
                "activation_mode": self.get_activation_mode(),
            },
        )
        return queued

    def _write_generated_code(self, spec: Dict[str, Any]) -> str:
        version = int(spec.get("version", 1))
        filename = f"{spec['name']}_v{version}.py"
        path = self.generated_dir / filename
        code = spec.get("code", "")
        path.write_text(code, encoding="utf-8")
        return str(path)

    def _publish_pending_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        spec = dict(spec)
        spec["status"] = "active"
        spec["updated_at"] = _now_iso()
        spec["code_path"] = self._write_generated_code(spec)
        spec.pop("code", None)

        catalog = self.state["tool_catalog"]
        old = catalog.get(spec["name"])
        if old and old.get("kind") == "generated":
            spec["version"] = int(old.get("version", 1)) + 1
            spec["code_path"] = self._write_generated_code(spec)

        catalog[spec["name"]] = spec
        self.state["tool_catalog_revision"] = int(self.state.get("tool_catalog_revision", 0)) + 1
        self._emit(
            "tool_activated",
            {
                "tool_id": spec["tool_id"],
                "name": spec["name"],
                "version": spec["version"],
                "activation_mode": self.get_activation_mode(),
            },
        )
        return spec

    def activate_pending(self, current_turn: int) -> Dict[str, Any]:
        pending = self.state.get("pending_tools", [])
        if not pending:
            return {"activated": [], "remaining": []}

        mode = self.get_activation_mode()
        keep: List[Dict[str, Any]] = []
        activated: List[Dict[str, Any]] = []

        for item in pending:
            eligible = int(item.get("eligible_turn", 0)) <= int(current_turn)
            approved = bool(item.get("approved", False))

            if not eligible:
                keep.append(item)
                continue

            if mode == "manual_review" and not approved:
                item["status"] = "awaiting_review"
                keep.append(item)
                continue

            activated.append(self._publish_pending_spec(item))

        self.state["pending_tools"] = keep
        return {
            "activated": [{"tool_id": x["tool_id"], "name": x["name"]} for x in activated],
            "remaining": [{"tool_id": x["tool_id"], "name": x["name"]} for x in keep],
        }

    def approve_pending(self, tool_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        target_ids = set(tool_ids or [])
        changed = 0
        for item in self.state.get("pending_tools", []):
            if target_ids and item.get("tool_id") not in target_ids:
                continue
            item["approved"] = True
            item["status"] = "approved"
            item["updated_at"] = _now_iso()
            changed += 1
        return {"approved": changed}

    def reject_pending(self, tool_ids: Optional[List[str]] = None, reason: str = "") -> Dict[str, Any]:
        target_ids = set(tool_ids or [])
        rejected: List[Dict[str, Any]] = []
        keep: List[Dict[str, Any]] = []
        for item in self.state.get("pending_tools", []):
            if target_ids and item.get("tool_id") not in target_ids:
                keep.append(item)
                continue
            if not target_ids:
                rejected.append(item)
            elif item.get("tool_id") in target_ids:
                rejected.append(item)
            else:
                keep.append(item)

        self.state["pending_tools"] = keep
        for item in rejected:
            self._emit(
                "tool_gate_failed",
                {
                    "tool_id": item.get("tool_id"),
                    "name": item.get("name"),
                    "reason": reason or "Rejected by user",
                },
            )

        return {"rejected": len(rejected)}

    def disable_generated(self, name_or_id: str) -> Dict[str, Any]:
        catalog = self.state.get("tool_catalog", {})
        for name, item in catalog.items():
            if name != name_or_id and item.get("tool_id") != name_or_id:
                continue
            if item.get("kind") != "generated":
                return {"ok": False, "error": "Builtin tools require human approval for changes."}
            item["status"] = "disabled"
            item["updated_at"] = _now_iso()
            self.state["tool_catalog_revision"] = int(self.state.get("tool_catalog_revision", 0)) + 1
            return {"ok": True, "name": name}
        return {"ok": False, "error": f"Tool not found: {name_or_id}"}

    def summary(self) -> Dict[str, Any]:
        catalog = self.state.get("tool_catalog", {})
        active = [x for x in catalog.values() if x.get("status") == "active"]
        pending = self.state.get("pending_tools", [])
        builtin = [x for x in active if x.get("kind") == "builtin"]
        generated = [x for x in active if x.get("kind") == "generated"]
        return {
            "revision": int(self.state.get("tool_catalog_revision", 0)),
            "activation_mode": self.get_activation_mode(),
            "active_builtin": len(builtin),
            "active_generated": len(generated),
            "pending_count": len(pending),
            "pending": pending,
            "active_tools": active,
        }

    def dump_json(self) -> str:
        return json.dumps(self.summary(), ensure_ascii=False, indent=2)
