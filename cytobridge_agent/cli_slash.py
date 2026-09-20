"""Slash command handling for terminal interactive sessions."""
from __future__ import annotations

import json
import shlex
from typing import Optional

from .session_controller import SessionController, format_session_listing
from .utils.llm_providers import get_llm_provider


QUIT_COMMANDS = {"exit", "quit", "q", "/exit", "/quit", "/q"}


def is_slash_command(message: str) -> bool:
    return message.strip().startswith("/")


def parse_slash_command(message: str) -> tuple[str, list[str], str]:
    text = message.strip()
    if not text.startswith("/"):
        return "", [], text
    body = text[1:].strip()
    if not body:
        return "help", [], ""
    try:
        parts = shlex.split(body)
    except ValueError:
        parts = body.split()
    command = parts[0].lower() if parts else "help"
    args = parts[1:]
    rest = body[len(parts[0]) :].strip() if parts else ""
    aliases = {
        "thinking": "reasoning",
        "think": "reasoning",
        "sessions": "session",
        "ls": "session",
        "new_session": "new",
        "stop_hook": "hooks",
    }
    return aliases.get(command, command), args, rest


def slash_help_message() -> str:
    return "\n".join(
        [
            "Slash commands:",
            "/help - show this help",
            "/new - start a fresh empty session",
            "/resume [SESSION_ID] - resume a saved session (opens picker when omitted)",
            "/resume --last - resume the latest saved session",
            "/resume --grep TEXT - resume from a filtered history picker",
            "/session [limit] - list saved sessions",
            "/status - show active session/model status",
            "/doctor - check local runtime health",
            "/perf [SESSION_ID] - summarize saved runtime events",
            "/jobs - list active runtime jobs",
            "/stop - request the active turn to stop",
            "/reasoning [off|low|medium|high|xhigh] - show or update thinking level",
            "/model [model_id] - show or switch model",
            "/provider [provider_id] [model_id] - show or switch provider",
            "/compact - compact the active session context",
            "/hooks [on|off|max N] - update stop hook settings for the active session",
            "/quit - leave interactive mode",
        ]
    )


def handle_cli_slash_command(controller: SessionController, message: str) -> Optional[str]:
    if not is_slash_command(message):
        return None
    command, args, _rest = parse_slash_command(message)

    if command in {"help", ""}:
        return slash_help_message()
    if command in {"quit", "exit", "q"}:
        raise EOFError
    if command == "status":
        return json.dumps(controller.status(), indent=2, ensure_ascii=False)
    if command == "doctor":
        from .runtime_diagnostics import format_doctor_report, run_doctor

        return format_doctor_report(run_doctor(store=controller.store, jobs=controller.jobs))
    if command == "perf":
        from .runtime_diagnostics import format_perf_report, session_perf_report

        session_id = args[0] if args else controller.status().get("session_id")
        if not session_id:
            return "No active session. Use /perf SESSION_ID."
        return format_perf_report(session_perf_report(str(session_id), store=controller.store))
    if command == "jobs":
        jobs = controller.jobs.list(active_only=True)
        if not jobs:
            return "No active runtime jobs."
        return "\n".join(
            f"{job.get('job_id')}  {job.get('kind')}  {job.get('status')}  "
            f"session={job.get('session_id') or '-'} pid={job.get('pid') or '-'} alive={job.get('pid_alive')}"
            for job in jobs
        )
    if command == "session":
        limit = 20
        query = None
        if args:
            try:
                limit = max(1, int(args[0]))
            except ValueError:
                query = " ".join(args).strip()
        from .cli_terminal import filter_sessions

        search_limit = max(limit, 200) if query else limit
        return format_session_listing(filter_sessions(controller.list_sessions(limit=search_limit), query=query)[:limit])
    if command == "new":
        controller.active_session = None
        controller.stop_requested = False
        return "Active session cleared. Provide a message to start a fresh session."
    if command == "resume":
        from .cli_runtime import SessionOpenSpec, resolve_resume_session_id

        try:
            query = None
            last = bool(args and args[0] == "--last")
            session_id_arg = None if not args or last else args[0]
            if args and args[0] == "--grep":
                query = " ".join(args[1:]).strip() or None
                session_id_arg = None
            session_id = resolve_resume_session_id(
                controller,
                SessionOpenSpec(
                    session_id=session_id_arg,
                    last=last,
                    picker=not args or bool(query),
                    picker_limit=20,
                    picker_prompt="Resume session",
                    query=query,
                ),
            )
        except RuntimeError as exc:
            return f"{exc} Usage: /resume SESSION_ID or /resume --last"
        if not session_id:
            return "Resume cancelled." if not args else "No saved session found."
        session = controller.resume_session(session_id)
        return f"Session resumed: {session.session_id}"
    if command == "stop":
        return controller.request_stop().get("message", "Stop requested.")
    if command == "compact":
        result = controller.compact_active_session()
        return json.dumps(result, indent=2, ensure_ascii=False)
    if command == "reasoning":
        if not args:
            return f"Current reasoning/thinking level: {controller.status().get('thinking_level') or 'low'}"
        result = controller.switch_model(thinking_level=args[0])
        return f"Reasoning/thinking level set to {result.get('thinking_level')}."
    if command == "model":
        if not args:
            return f"Current model: {controller.status().get('model') or 'unknown'}"
        result = controller.switch_model(model=" ".join(args).strip())
        return f"Model set to {result.get('model')}."
    if command == "provider":
        if not args:
            return f"Current provider: {controller.status().get('provider') or 'auto'}"
        provider = args[0]
        model = " ".join(args[1:]).strip() or get_llm_provider(provider).default_model or None
        result = controller.switch_model(provider=provider, model=model)
        suffix = f", model {result.get('model')}" if result.get("model") else ""
        return f"Provider set to {result.get('provider')}{suffix}."
    if command == "hooks":
        session = controller.active_session
        if session is None:
            return "Session not initialized."
        if not args:
            return json.dumps(
                {
                    "stop_hook_enabled": session.state.get("stop_hook_enabled"),
                    "stop_hook_mode": session.state.get("stop_hook_mode"),
                    "stop_hook_max_triggers": session.state.get("stop_hook_max_triggers"),
                },
                indent=2,
                ensure_ascii=False,
            )
        idx = 0
        while idx < len(args):
            token = args[idx].lower()
            if token == "on":
                session.state["stop_hook_enabled"] = True
            elif token == "off":
                session.state["stop_hook_enabled"] = False
            elif token == "max" and idx + 1 < len(args):
                session.state["stop_hook_max_triggers"] = int(args[idx + 1])
                idx += 1
            elif token.startswith("max="):
                session.state["stop_hook_max_triggers"] = int(token.split("=", 1)[1])
            idx += 1
        session.planner.refresh_tooling()
        session._save_conversation()
        return "Stop hook settings updated."
    return f"Unknown slash command: /{command}. Use /help."
