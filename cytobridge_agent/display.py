"""
Display Manager for CytoBridge.

Handles all console output styling using the rich library.
Provides a unified interface for agents to print thoughts, code, and status.
"""
from __future__ import annotations

import contextlib
import json
import threading
import uuid
from typing import Optional, Any
from datetime import datetime
from pathlib import Path

# Try accessing rich, fallback provided if missing
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.theme import Theme
    from rich.status import Status
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Any 
    Panel = Any
    Markdown = Any
    Syntax = Any
    Status = Any

# Custom theme
CYTO_THEME = {
    "agent.name": "bold green",
    "agent.text": "green",
    "user.name": "bold yellow",
    "user.text": "yellow",
    "tool.name": "dim cyan",
    "tool.args": "dim",
    "result.header": "bold white on green",
    "error": "bold white on red"
}

class DisplayManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DisplayManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance


    def _init(self):
        """Initialize the console."""
        if RICH_AVAILABLE:
            self.console = Console(theme=Theme(CYTO_THEME))
        else:
            self.console = None
        self.event_callback = None
        self._local = threading.local()

    def set_event_callback(self, callback):
        """Set a callback function to receive display events."""
        self.event_callback = callback

    def _event_context(self) -> dict:
        context = getattr(self._local, "event_context", None)
        return dict(context or {})

    @contextlib.contextmanager
    def event_context(self, **metadata):
        """Attach metadata such as turn_id to events emitted in this thread."""
        previous = self._event_context()
        current = {**previous, **{k: v for k, v in metadata.items() if v}}
        self._local.event_context = current
        try:
            yield
        finally:
            self._local.event_context = previous

    def _tool_stack(self) -> list[str]:
        stack = getattr(self._local, "tool_call_stack", None)
        if stack is None:
            stack = []
            self._local.tool_call_stack = stack
        return stack

    def finish_tool_call(self, tool_call_id: Optional[str] = None) -> None:
        """Clear a tool call from the thread-local display stack."""
        stack = self._tool_stack()
        if not stack:
            return
        target = str(tool_call_id or stack[-1])
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx] == target:
                del stack[idx]
                return

    @classmethod
    def _to_json_safe(cls, value: Any) -> Any:
        """Recursively convert arbitrary payloads into JSON-serializable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): cls._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return cls._to_json_safe(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return cls._to_json_safe(value.dict())
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return cls._to_json_safe(vars(value))
            except Exception:
                pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    def _emit(self, event_type: str, data: dict):
        """Emit an event to the callback if registered."""
        if self.event_callback:
            context = self._event_context()
            payload = {
                "type": event_type,
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "data": self._to_json_safe(data),
            }
            if context.get("turn_id"):
                payload["turn_id"] = str(context["turn_id"])
            if context.get("process_id"):
                payload["process_id"] = str(context["process_id"])
            self.event_callback(payload)

    def print_welcome(self):
        """Print the application header."""
        if RICH_AVAILABLE:
            self.console.print(Panel.fit(
                "[bold cyan]🧬 CytoBridge Analysis Agent[/]\n"
                "[dim]Advanced Single-Cell Fate & Dynamics Analysis[/]",
                border_style="cyan",
                padding=(1, 4)
            ))
        else:
            print("="*40)
            print("  CytoBridge Analysis Agent")
            print("="*40)
        self._emit("welcome", {})

    def print_user_message(self, content: str):
        """Print a message from the user."""
        if RICH_AVAILABLE:
            self.console.print()
            self.console.print(Text("You:", style="user.name"))
            self.console.print(Text(str(content), style="user.text"))
        else:
            print(f"\nYou: {content}")
        self._emit("user_message", {"content": content})

    def print_agent_thought(self, agent_name: str, content: str):
        """
        Print an agent's internal thought process.
        
        Args:
            agent_name: Name of the agent (e.g., "PreprocessingAgent").
            content: The thought content.
        """
        if not content or not content.strip():
            return
            
        time_str = datetime.now().strftime("%H:%M:%S")
        title = f"🤖 {agent_name} ({time_str})"
        
        if RICH_AVAILABLE:
            self.console.print(Panel(
                Markdown(content),
                title=title,
                border_style="blue",
                title_align="left"
            ))
        else:
            print(f"\n[{title}]\n{content}\n")
        
        self._emit("agent_thought", {"agent": agent_name, "content": content})

    def print_tool_start(self, tool_name: str, args: dict, reason: str = None, tool_call_id: Optional[str] = None) -> str:
        """Print the start of a tool execution."""
        call_id = str(tool_call_id or uuid.uuid4())
        self._tool_stack().append(call_id)
        if RICH_AVAILABLE:
            args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "..."
            
            text = Text()
            text.append("🛠️  Running Tool: ", style="bold cyan")
            text.append(tool_name, style="bold cyan")
            if reason:
                text.append(f"\n    Reason: ", style="dim yellow")
                text.append(reason, style="yellow")
            text.append("\n    Args: ", style="dim")
            text.append(args_str, style="dim")
            self.console.print(text)
        else:
            print(f"🛠️  Running Tool: {tool_name} | Args: {args}")
        
        self._emit("tool_start", {"tool": tool_name, "args": args, "reason": reason, "tool_call_id": call_id})
        return call_id

    def print_tool_output(self, content: str, tool_call_id: Optional[str] = None):
        """Print the output of a tool."""
        if not content:
            return
        stack = self._tool_stack()
        call_id = str(tool_call_id or (stack[-1] if stack else ""))

        if len(content) > 1000:
            display_content = content[:1000] + "\n... (truncated)"
        else:
            display_content = content

        if RICH_AVAILABLE:
            self.console.print(Panel(
                display_content,
                title="📋 Tool Result",
                border_style="green",
                style="dim"
            ))
        else:
            print(f"📋 Result:\n{display_content}\n")
        
        payload = {"content": content, "display_content": display_content}
        if call_id:
            payload["tool_call_id"] = call_id
        self._emit("tool_output", payload)

    def print_code(self, code: str, language: str = "python"):
        """Print a block of code with syntax highlighting."""
        if RICH_AVAILABLE:
            self.console.print(Syntax(
                code, 
                language, 
                theme="monokai", 
                line_numbers=True,
                word_wrap=True
            ))
        else:
            print(f"```python\n{code}\n```")
        
        self._emit("code", {"code": code, "language": language})

    def print_error(self, message: str):
        """Print an error message."""
        if RICH_AVAILABLE:
            self.console.print(Text(f"❌ Error: {message}", style="error"))
        else:
            print(f"❌ Error: {message}")
        
        self._emit("error", {"message": message})

    def print_status(self, message: str):
        """Print a status message."""
        if RICH_AVAILABLE:
            self.console.print(Text(f"⏳ {message}", style="dim cyan"))
        else:
            print(f"⏳ {message}")
        
        self._emit("status", {"message": message})

    @contextlib.contextmanager
    def spinner(self, message: str):
        """Show a spinner for long-running operations."""
        self._emit("spinner_start", {"message": message})
        try:
            if RICH_AVAILABLE:
                with self.console.status(Text(str(message), style="bold cyan"), spinner="dots") as status:
                    yield status
            else:
                print(f"⏳ {message}...")
                yield
        finally:
            self._emit("spinner_end", {})

    def print_result_header(self, title: str):
         """Print a header for a major result."""
         if RICH_AVAILABLE:
             self.console.rule(Text(str(title), style="bold green"))
         else:
             print(f"\n--- {title} ---")
         self._emit("result_header", {"title": title})

    def print_theory_selection(self, model_family: str, reasoning: str, theory_support: str = None, config: dict = None):
        """Print and broadcast the theory agent's model selection with reasoning."""
        if RICH_AVAILABLE:
            from rich.panel import Panel
            content = f"**Model:** {model_family}\n\n**Reasoning:** {reasoning}"
            if theory_support:
                content += f"\n\n**Theory Support:** {theory_support}"
            self.console.print(Panel(
                Markdown(content),
                title="🎯 Theory Selection",
                border_style="magenta",
                title_align="left"
            ))
        else:
            print(f"\n🎯 Theory Selection: {model_family}")
            print(f"   Reasoning: {reasoning}")
        
        self._emit("theory_selection", {
            "model_family": model_family,
            "reasoning": reasoning,
            "theory_support": theory_support or "",
            "config": config or {}
        })

    def print_stage(self, stage: str, description: str = ""):
        """Print and broadcast a major pipeline stage change."""
        stage_icons = {
            "preprocessing": "📊",
            "theory_selection": "🔬",
            "training": "🎯",
            "downstream": "📈",
            "report": "📝",
            "complete": "✅"
        }
        icon = stage_icons.get(stage.lower(), "⏳")
        
        if RICH_AVAILABLE:
            from rich.panel import Panel
            self.console.print(Panel(
                Text(str(description or stage.title()), style="bold"),
                title=f"{icon} {stage.upper()}",
                border_style="cyan",
                title_align="left"
            ))
        else:
            print(f"\n{icon} === {stage.upper()} === {description}")
        
        self._emit("stage", {
            "stage": stage,
            "description": description or stage.title()
        })

    def print_progress(self, message: str, progress: float = None, progress_id: str = "training"):
        """Print training or processing progress updates.
        
        Args:
            message: Progress message to display.
            progress: Float 0-1 representing progress percentage.
            progress_id: Unique ID for this progress bar (allows in-place update).
        """
        # For console: use carriage return to update in place
        if RICH_AVAILABLE:
            if progress is not None:
                # Use \r to overwrite same line in terminal
                import sys
                sys.stdout.write(f"\r📈 [{progress*100:.1f}%] {message}    ")
                sys.stdout.flush()
                if progress >= 1.0:
                    print()  # Newline when complete
            else:
                self.console.print(Text(f"📈 {message}", style="dim cyan"))
        else:
            if progress is not None:
                import sys
                sys.stdout.write(f"\r📈 [{progress*100:.1f}%] {message}    ")
                sys.stdout.flush()
                if progress >= 1.0:
                    print()
            else:
                print(f"📈 {message}")
        
        # Emit event with ID so frontend can UPDATE existing progress bar
        self._emit("training_progress", {
            "id": progress_id,
            "message": message,
            "progress": progress
        })
