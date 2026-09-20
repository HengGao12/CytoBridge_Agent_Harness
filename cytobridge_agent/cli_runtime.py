"""Shared runtime helpers for CellCompass CLI entrypoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .runtime_events import runtime_event
from .session_controller import SessionController, print_jsonl_event

if TYPE_CHECKING:
    from .interactive import InteractiveSession


@dataclass
class SessionOpenSpec:
    """Declarative session-open request shared by CLI commands."""

    input_path: Optional[str] = None
    question: str = ""
    output_dir: Optional[str] = None
    device: str = "cuda"
    report_format: str = "html"
    enable_multimodal: bool = True
    session_id: Optional[str] = None
    last: bool = False
    require_resume: bool = False
    picker: bool = False
    picker_limit: int = 20
    picker_prompt: str = "Resume session"
    query: Optional[str] = None
    cwd: Optional[str] = None
    user_goal_overrides: Optional[Dict[str, Any]] = None


class CLIEventSink:
    """Small adapter for terminal/JSON/TUI event output.

    It is intentionally minimal: the long-running agent still owns detailed
    tool events, while CLI entrypoints share the same session/turn envelope.
    """

    def __init__(self, *, json_enabled: bool = False, rich: bool = False) -> None:
        self.json_enabled = bool(json_enabled)
        self.rich = bool(rich)
        self.events = []
        self._console = None
        if self.rich:
            try:
                from rich.console import Console

                self._console = Console()
            except Exception:
                self._console = None

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        data = dict(payload or {})
        event = runtime_event(event_type, data, session_id=data.get("session_id"), source="cli")
        self.events.append(event)
        if self.json_enabled:
            print_jsonl_event(event["type"], {**data, "event_id": event["event_id"], "timestamp": event["timestamp"]})
            return
        if self._console is not None:
            if event_type in {"session_started", "session_resumed"}:
                label = "resumed" if event_type == "session_resumed" else "started"
                self._console.print(f"[bold cyan]session {label}[/]: {data.get('session_id')}")
            elif event_type == "turn_complete":
                self._console.print(f"[bold green]turn complete[/]: {data.get('session_id')}")
            elif event_type == "error":
                self._console.print(f"[bold red]error[/]: {data.get('message')}")
            return


def user_goal_from_spec(spec: SessionOpenSpec) -> Dict[str, Any]:
    goal = {
        "raw_question": spec.question,
        "requested_analyses": [],
        "device": spec.device or "cuda",
        "report_format": spec.report_format or "html",
    }
    if spec.user_goal_overrides:
        goal.update(dict(spec.user_goal_overrides))
    return goal


def resolve_resume_session_id(controller: SessionController, spec: SessionOpenSpec) -> Optional[str]:
    """Resolve an explicit, latest, or interactively selected session id."""
    session_id = spec.session_id
    if spec.query or spec.cwd:
        from .cli_terminal import filter_sessions

        search_limit = max(int(spec.picker_limit or 20), 200)
        candidates = filter_sessions(
            controller.list_sessions(limit=search_limit),
            query=spec.query,
            cwd=spec.cwd,
        )[: max(1, int(spec.picker_limit or 20))]
        if spec.last:
            return str(candidates[0].get("session_id") or "") if candidates else None
        if spec.picker:
            from .cli_terminal import choose_session_interactively

            return choose_session_interactively(candidates, prompt=spec.picker_prompt)
    if spec.last:
        latest = controller.latest_session_id()
        if latest or not spec.picker:
            return latest
    if session_id:
        return session_id
    if not spec.picker:
        return None

    from .cli_terminal import choose_session_interactively

    return choose_session_interactively(
        controller.list_sessions(limit=max(1, int(spec.picker_limit or 20))),
        prompt=spec.picker_prompt,
    )


def open_or_create_session(
    controller: SessionController,
    spec: SessionOpenSpec,
) -> Tuple["InteractiveSession", bool]:
    """Resume an existing session when requested, otherwise create a new one."""
    session_id = resolve_resume_session_id(controller, spec)
    if session_id:
        return controller.resume_session(session_id), True
    if spec.require_resume or spec.last:
        raise RuntimeError("No saved session found.")
    return (
        controller.new_session(
            input_path=spec.input_path,
            user_goal=user_goal_from_spec(spec),
            output_dir=spec.output_dir,
            enable_multimodal=spec.enable_multimodal,
        ),
        False,
    )


def emit_session_open_event(
    *,
    json_enabled: bool,
    session: "InteractiveSession",
    resumed: bool,
    sink: Optional[CLIEventSink] = None,
) -> None:
    (sink or CLIEventSink(json_enabled=json_enabled)).emit(
        "session_resumed" if resumed else "session_started",
        {"session_id": session.session_id},
    )
