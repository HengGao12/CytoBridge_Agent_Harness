"""Rich-based terminal UI for CellCompass sessions."""
from __future__ import annotations

from typing import Optional

from .cli_runtime import CLIEventSink, SessionOpenSpec, open_or_create_session
from .display import DisplayManager


def _console():
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


def run_tui(controller, spec: SessionOpenSpec) -> int:
    """Run an interactive terminal UI without adding a heavyweight TUI dependency."""
    from .cli_slash import QUIT_COMMANDS, handle_cli_slash_command, is_slash_command
    from .cli_terminal import enable_readline_history

    console = _console()
    sink = CLIEventSink(rich=True)
    try:
        session, resumed = open_or_create_session(controller, spec)
    except RuntimeError as exc:
        message = f"{exc} Use: cellcompass tui --resume SESSION_ID or cellcompass tui --last"
        if console:
            console.print(f"[bold red]{message}[/]")
        else:
            print(message)
        return 1

    if console:
        from rich.panel import Panel
        from rich.table import Table

        table = Table.grid(padding=(0, 2))
        table.add_row("Session", session.session_id)
        table.add_row("Mode", "resumed" if resumed else "new")
        table.add_row("Data", str(session.input_path or "(not set)"))
        table.add_row("Output", str(session.output_dir or "(not set)"))
        console.print(Panel(table, title="CellCompass", border_style="cyan"))
        console.print("[dim]Use /help for commands, /resume to switch session, /quit to exit.[/]")
    else:
        print("=" * 60)
        print("CellCompass")
        print("=" * 60)
        print(f"Session: {session.session_id}")
        print("Use /help for commands, /quit to exit.")

    initial = str(spec.question or "").strip()
    if initial and not resumed:
        _run_tui_turn(controller, initial, sink=sink, console=console)

    enable_readline_history()
    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in QUIT_COMMANDS:
                return 0
            if is_slash_command(user_input):
                try:
                    output = handle_cli_slash_command(controller, user_input)
                except EOFError:
                    return 0
                if output:
                    if console:
                        console.print(output)
                    else:
                        print(output)
                continue
            _run_tui_turn(controller, user_input, sink=sink, console=console)
        except KeyboardInterrupt:
            if console:
                console.print("[yellow]Interrupted. Session checkpoint is preserved when available.[/]")
            else:
                print("\nInterrupted. Session checkpoint is preserved when available.")
            return 130
        except EOFError:
            return 0


def _run_tui_turn(controller, prompt: str, *, sink: CLIEventSink, console: Optional[object]) -> None:
    previous_callback = DisplayManager().event_callback

    def on_event(event):
        event_type = str((event or {}).get("type") or "event")
        data = dict((event or {}).get("data") or {})
        sink.emit(event_type, data)
        if console and event_type in {"agent_thought", "tool_start", "tool_end", "tool_error", "error"}:
            label = event_type.replace("_", " ")
            message = data.get("content") or data.get("tool") or data.get("tool_name") or data.get("message") or ""
            console.print(f"[dim]{label}[/] {message}")

    DisplayManager().set_event_callback(on_event)
    if console:
        try:
            with console.status("Agent is working...", spinner="dots"):
                result = controller.run_turn(prompt)
            console.print("\n[bold green]Agent[/]")
            console.print(result.response)
        finally:
            DisplayManager().set_event_callback(previous_callback)
    else:
        try:
            print("Agent is working...")
            result = controller.run_turn(prompt)
            print(f"\nAgent > {result.response}")
        finally:
            DisplayManager().set_event_callback(previous_callback)
    sink.emit("turn_complete", {"session_id": result.session_id})
