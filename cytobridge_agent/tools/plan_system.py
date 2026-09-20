"""Shared lightweight planning system for planner/downstream agents."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


PLAN_STATUSES = {"pending", "in_progress", "completed"}
STATUS_ALIASES = {
    "todo": "pending",
    "to_do": "pending",
    "not_started": "pending",
    "notstarted": "pending",
    "waiting": "pending",
    "inprogress": "in_progress",
    "in-progress": "in_progress",
    "doing": "in_progress",
    "active": "in_progress",
    "current": "in_progress",
    "done": "completed",
    "finished": "completed",
    "complete": "completed",
}


EventSink = Optional[Callable[[str, Dict[str, Any]], None]]


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _coerce_plan_item_object(item: Any) -> Any:
    """Best-effort conversion for pydantic/dataclass-like plan items to dict."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump()
        except Exception:
            pass
    if hasattr(item, "dict"):
        try:
            return item.dict()
        except Exception:
            pass
    return item


def _normalize_status(raw: Any) -> str:
    token = str(raw or "pending").strip().lower().replace(" ", "_")
    token = STATUS_ALIASES.get(token, token)
    return token


def normalize_plan_items(plan: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """Validate and normalize plan items."""
    if not isinstance(plan, list) or not plan:
        return None, "Error: plan must be a non-empty list."

    normalized: List[Dict[str, str]] = []
    in_progress_count = 0

    for idx, raw_item in enumerate(plan, start=1):
        item = _coerce_plan_item_object(raw_item)
        if not isinstance(item, dict):
            return None, f"Error: plan item #{idx} must be an object."

        step = str(item.get("step", "") or item.get("title", "") or item.get("task", "")).strip()
        if not step:
            return None, f"Error: plan item #{idx} missing non-empty 'step'."

        status = _normalize_status(item.get("status", "pending"))
        if status not in PLAN_STATUSES:
            return None, (
                f"Error: plan item #{idx} has invalid status '{status}'. "
                "Allowed: pending, in_progress, completed."
            )

        if status == "in_progress":
            in_progress_count += 1

        normalized.append({"step": step, "status": status})

    if in_progress_count > 1:
        return None, "Error: at most one plan item can be in_progress."

    # Codex-style invariant: if there are unfinished steps, keep one active pointer.
    if in_progress_count == 0:
        first_pending_idx = next((i for i, it in enumerate(normalized) if it.get("status") == "pending"), None)
        if first_pending_idx is not None:
            normalized[first_pending_idx]["status"] = "in_progress"

    return normalized, None


def extract_plan_items_from_text(plan_text: str) -> List[Dict[str, str]]:
    """Extract plan steps from numbered/bulleted text."""
    if not plan_text:
        return []

    # 1) Prefer strict JSON plan formats when available.
    json_obj: Optional[Any] = None
    text = plan_text.strip()
    if text:
        try:
            json_obj = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    json_obj = json.loads(text[start : end + 1])
                except Exception:
                    json_obj = None

    if isinstance(json_obj, dict):
        raw_items = json_obj.get("steps") or json_obj.get("items")
        if isinstance(raw_items, list):
            normalized: List[Dict[str, str]] = []
            for raw in raw_items:
                if isinstance(raw, str) and raw.strip():
                    normalized.append({"step": raw.strip(), "status": "pending"})
                elif isinstance(raw, dict):
                    step = str(raw.get("step", "")).strip()
                    if not step:
                        continue
                    status = _normalize_status(raw.get("status", "pending"))
                    if status not in PLAN_STATUSES:
                        status = "pending"
                    normalized.append({"step": step, "status": status})
            if normalized:
                return normalized

    if isinstance(json_obj, list):
        normalized: List[Dict[str, str]] = []
        for raw in json_obj:
            if isinstance(raw, str) and raw.strip():
                normalized.append({"step": raw.strip(), "status": "pending"})
            elif isinstance(raw, dict):
                step = str(raw.get("step", "")).strip()
                if not step:
                    continue
                status = _normalize_status(raw.get("status", "pending"))
                if status not in PLAN_STATUSES:
                    status = "pending"
                normalized.append({"step": step, "status": status})
        if normalized:
            return normalized

    # 2) Fallback to textual extraction.
    items: List[Dict[str, str]] = []
    lines = [line.strip() for line in plan_text.splitlines() if line.strip()]
    pattern = re.compile(r"^(?:\d+[.)]|[-*])\s+(.*\S)\s*$")
    for line in lines:
        match = pattern.match(line)
        if match:
            step = match.group(1).strip()
            if step:
                items.append({"step": step, "status": "pending"})

    if items:
        # Prefer high-level headings if model produced dense numbered sub-steps.
        heading_items = [i for i in items if i["step"].startswith("**") and i["step"].endswith("**")]
        if heading_items:
            return heading_items
        return items

    return [{"step": line, "status": "pending"} for line in lines]


def build_plan_state_from_text(plan_text: str, explanation: str = "") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build normalized plan state from free text."""
    items = extract_plan_items_from_text(plan_text)
    normalized, err = normalize_plan_items(items)
    if err:
        return None, err
    return {
        "items": normalized,
        "explanation": explanation.strip(),
        "updated_at": _now_iso(),
    }, None


def build_plan_state_from_items(
    plan: List[Dict[str, Any]],
    explanation: str = "",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build normalized plan state from structured items."""
    normalized, err = normalize_plan_items(plan)
    if err:
        return None, err
    return {
        "items": normalized,
        "explanation": explanation.strip(),
        "updated_at": _now_iso(),
    }, None


def resolve_numeric_step_references(
    plan: List[Dict[str, Any]],
    previous_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map numeric `step` references (1-based index) back to prior step text."""
    if not plan or not previous_items:
        return plan

    resolved: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(plan, start=1):
        item = _coerce_plan_item_object(raw_item)
        if not isinstance(item, dict):
            resolved.append(item)
            continue
        step_raw = item.get("step")
        step_idx: Optional[int] = None
        if isinstance(step_raw, int):
            step_idx = step_raw
        elif isinstance(step_raw, str) and step_raw.strip().isdigit():
            try:
                step_idx = int(step_raw.strip())
            except Exception:
                step_idx = None

        if step_idx is None:
            resolved.append(item)
            continue

        if 1 <= step_idx <= len(previous_items):
            prev_text = str(previous_items[step_idx - 1].get("step", "")).strip()
            if prev_text:
                repaired = dict(item)
                repaired["step"] = prev_text
                resolved.append(repaired)
                continue

        resolved.append(item)
    return resolved


def resolve_missing_step_texts(
    plan: List[Dict[str, Any]],
    previous_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Repair missing `step` text by index from previous plan when possible."""
    if not plan or not previous_items:
        return plan

    resolved: List[Dict[str, Any]] = []
    for idx, raw_item in enumerate(plan):
        item = _coerce_plan_item_object(raw_item)
        if not isinstance(item, dict):
            resolved.append(item)
            continue
        step = str(item.get("step", "") or item.get("title", "") or item.get("task", "")).strip()
        if step:
            resolved.append(item)
            continue
        if idx < len(previous_items):
            prev_text = str(previous_items[idx].get("step", "")).strip()
            if prev_text:
                repaired = dict(item)
                repaired["step"] = prev_text
                resolved.append(repaired)
                continue
        resolved.append(item)
    return resolved


def render_plan_state(plan_state: Dict[str, Any]) -> str:
    """Render plan state into readable text."""
    items = list(plan_state.get("items") or [])
    explanation = str(plan_state.get("explanation", "")).strip()
    updated_at = str(plan_state.get("updated_at", "")).strip()

    if not items:
        return "No active plan."

    completed = sum(1 for item in items if item.get("status") == "completed")
    in_progress = sum(1 for item in items if item.get("status") == "in_progress")
    total = len(items)

    lines = [f"Plan progress: {completed}/{total} completed, {in_progress} in progress"]
    if explanation:
        lines.append(f"Explanation: {explanation}")
    if updated_at:
        lines.append(f"Updated at: {updated_at}")
    lines.append("Steps:")

    for idx, item in enumerate(items, start=1):
        step = str(item.get("step", "")).strip()
        status = str(item.get("status", "pending")).strip()
        lines.append(f"{idx}. [{status}] {step}")

    return "\n".join(lines)


class PlanService:
    """Shared plan facade over ``AgentState`` fields with optional event emission."""

    def __init__(
        self,
        shared_state: Dict[str, Any],
        event_sink: EventSink = None,
        namespace: str = "planner",
    ):
        self.state = shared_state
        self.event_sink = event_sink
        ns = str(namespace or "planner").strip().lower()
        if ns == "downstream":
            self.scope = "downstream"
            self.keys = {
                "items": "downstream_execution_plan_items",
                "explanation": "downstream_execution_plan_explanation",
                "updated_at": "downstream_execution_plan_updated_at",
                "rendered": "downstream_execution_plan",
                "draft": "downstream_plan_draft",
            }
        else:
            self.scope = "planner"
            self.keys = {
                "items": "execution_plan_items",
                "explanation": "execution_plan_explanation",
                "updated_at": "execution_plan_updated_at",
                "rendered": "execution_plan",
                "draft": "planner_mode_plan_draft",
            }

    def get(self) -> Dict[str, Any]:
        return {
            "items": list(self.state.get(self.keys["items"], [])),
            "explanation": str(self.state.get(self.keys["explanation"], "")),
            "updated_at": str(self.state.get(self.keys["updated_at"], "")),
        }

    def set(self, plan_state: Dict[str, Any], source: str = "update_plan") -> Dict[str, Any]:
        self.state[self.keys["items"]] = list(plan_state.get("items", []))
        self.state[self.keys["explanation"]] = str(plan_state.get("explanation", ""))
        self.state[self.keys["updated_at"]] = str(plan_state.get("updated_at", ""))
        self.state[self.keys["rendered"]] = render_plan_state(plan_state)

        if self.event_sink:
            self.event_sink(
                "turn_plan_updated",
                {
                    "source": source,
                    "scope": self.scope,
                    "plan": list(plan_state.get("items", [])),
                    "explanation": str(plan_state.get("explanation", "")),
                    "updated_at": str(plan_state.get("updated_at", "")),
                },
            )
        return plan_state

    def set_from_text(self, plan_text: str, explanation: str = "", source: str = "set_plan_from_text") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        self.state[self.keys["draft"]] = str(plan_text or "")
        plan_state, err = build_plan_state_from_text(plan_text, explanation=explanation)
        if err:
            return None, err
        assert plan_state is not None
        self.set(plan_state, source=source)
        return plan_state, None

    def update(self, plan: List[Dict[str, Any]], explanation: str = "", source: str = "update_plan") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        previous_items = list(self.state.get(self.keys["items"], []))
        repaired_plan = resolve_numeric_step_references(
            plan,
            previous_items,
        )
        repaired_plan = resolve_missing_step_texts(
            repaired_plan,
            previous_items,
        )
        plan_state, err = build_plan_state_from_items(repaired_plan, explanation=explanation)
        if err:
            return None, err
        assert plan_state is not None
        self.set(plan_state, source=source)
        return plan_state, None

    def render(self) -> str:
        return render_plan_state(self.get())
