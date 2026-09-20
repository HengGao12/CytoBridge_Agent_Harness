"""Interactive CLI session orchestration with checkpoint/resume support."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.load import dumpd, load
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from .display import DisplayManager
from .runtime_v2 import CytoBridgeAgent
from .runtime_v2.state import ensure_runtime_v2_state
from .schemas import UserGoal, create_initial_state
from .tools.path_resolver import pick_path_from_message
from .tools.turn_phase import strip_phase_tag
from .utils.llm_factory import instantiate_llm


class InteractiveSession:
    """Manages a multi-turn interactive CytoBridge session."""

    def __init__(
        self,
        input_path: Optional[str],
        user_goal: Dict[str, Any],
        llm: BaseChatModel,
        output_dir: Optional[str] = None,
        enable_multimodal: bool = True,
    ):
        from .conversation_store import ConversationStore

        self.input_path = input_path
        self.llm = llm
        self.output_dir = output_dir
        self.enable_multimodal = enable_multimodal
        self.display = DisplayManager()
        self._runtime_lock = threading.Lock()

        self.store = ConversationStore()
        self.session_id = self.store.generate_session_id()

        ug_obj = UserGoal(**user_goal)
        self.state = create_initial_state(input_path, ug_obj)
        self.state["session_id"] = self.session_id
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            self.state["output_dir"] = output_dir
        self.state["enable_multimodal"] = self.enable_multimodal

        self.local_events_log: List[Dict[str, Any]] = []
        self.resume_diagnostics: Dict[str, Any] = {
            "source": "snapshot",
            "migrated_from_v1": False,
            "degraded": False,
            "warnings": [],
        }

        self.agent = CytoBridgeAgent(
            llm,
            self.state,
            checkpoint_callback=self._save_checkpoint,
            event_callback=self._on_agent_event,
        )
        self.planner = self.agent
        self.conversation_history: List[Dict[str, str]] = []

    @staticmethod
    def _messages_checkpoint_aligned(
        checkpoint_messages: List[BaseMessage],
        history_messages: List[BaseMessage],
    ) -> bool:
        if not checkpoint_messages or not history_messages:
            return False
        if len(checkpoint_messages) != len(history_messages):
            return False
        checkpoint_last = checkpoint_messages[-1]
        history_last = history_messages[-1]
        return (
            type(checkpoint_last) is type(history_last)
            and str(getattr(checkpoint_last, "id", "") or "") == str(getattr(history_last, "id", "") or "")
            and str(getattr(checkpoint_last, "content", "") or "") == str(getattr(history_last, "content", "") or "")
        )

    def _on_agent_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._record_timeline_event(event_type, payload)

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime

        return datetime.now().isoformat()

    @classmethod
    def from_checkpoint(cls, session_id: str, llm: BaseChatModel) -> "InteractiveSession":
        from .conversation_store import ConversationStore

        store = ConversationStore()
        data = store.get_conversation(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        agent_state = data.get("agent_state") or {}
        metadata = data.get("metadata", {})

        session = cls(
            input_path=metadata.get("input_path") or agent_state.get("input_path"),
            user_goal=agent_state.get("user_goal", {}),
            llm=llm,
            output_dir=metadata.get("output_dir"),
            enable_multimodal=bool(agent_state.get("enable_multimodal", True)),
        )
        session.session_id = session_id
        session.store = store

        session.state.update(agent_state)
        ensure_runtime_v2_state(session.state)
        session.state["session_id"] = session_id
        session.input_path = session.state.get("input_path")
        session.planner.set_thread_id(session_id)
        session.planner.refresh_tooling()

        session.conversation_history = list(data.get("messages", []))
        session.local_events_log = list(data.get("events_log", []))
        session.resume_diagnostics = dict(data.get("resume_diagnostics") or session.resume_diagnostics)

        snapshots = data.get("agent_snapshots") or {}
        histories = data.get("agent_histories") or {}
        planner_snapshot = dict(snapshots.get("planner") or {})
        restored_checkpoint = session.planner.restore_runtime_snapshot(planner_snapshot)
        checkpoint_messages = session.planner.checkpoint_messages() if restored_checkpoint else []
        history_messages: List[BaseMessage] = []
        if "planner" in histories:
            history_messages = [load(msg) for msg in histories["planner"]]

        if history_messages:
            session.planner.chat_history = history_messages
            if not cls._messages_checkpoint_aligned(checkpoint_messages, history_messages):
                session.planner.replace_chat_history(history_messages, reset_checkpoint=True)
        elif checkpoint_messages:
            session.planner.chat_history = checkpoint_messages
        if not snapshots and histories:
            snapshots = {
                name: {"chat_history": hist}
                for name, hist in histories.items()
                if name != "planner"
            }

        restore_diag = session._rehydrate_sub_agents(snapshots, histories)
        if restore_diag.get("degraded"):
            session.resume_diagnostics["degraded"] = True
            session.resume_diagnostics.setdefault("warnings", []).extend(restore_diag.get("reasons", []))

        session._sync_adata_manager_with_state()
        return session

    def update_llm(self, llm: BaseChatModel) -> Dict[str, Any]:
        """Hot-swap model while preserving planner/sub-agent histories."""
        snapshots = self._collect_agent_snapshots()
        histories = self._collect_agent_histories()
        planner_history = list(self.planner.chat_history)
        old_stop_check = self.planner.stop_check

        self.llm = llm
        self.agent = CytoBridgeAgent(
            llm,
            self.state,
            checkpoint_callback=self._save_checkpoint,
            event_callback=self._on_agent_event,
        )
        self.planner = self.agent
        self.planner.chat_history = planner_history
        self.planner.stop_check = old_stop_check
        self.planner.set_thread_id(str(self.state.get("session_id") or self.session_id))
        self.planner.restore_runtime_snapshot((snapshots.get("planner") or {}))

        restore_diag = self._rehydrate_sub_agents(snapshots, histories)
        if restore_diag.get("degraded"):
            self.resume_diagnostics["degraded"] = True
            self.resume_diagnostics.setdefault("warnings", []).extend(restore_diag.get("reasons", []))
        self._save_checkpoint()
        return restore_diag

    def _sync_adata_manager_with_state(self) -> None:
        from .tools.adata_manager import AnnDataManager

        manager = AnnDataManager()
        target = None
        final_config = self.state.get("final_config") or {}
        if isinstance(final_config, dict):
            target = final_config.get("path")
        target = target or self.state.get("preprocessed_path") or self.state.get("input_path")

        if not target:
            return

        target_path = Path(target).expanduser().resolve()
        if not target_path.exists():
            self.resume_diagnostics["degraded"] = True
            self.resume_diagnostics.setdefault("warnings", []).append(
                f"Data path missing during resume: {target_path}"
            )
            return

        current = manager.get_path()
        if current and Path(current).expanduser().resolve() == target_path:
            return

        manager.bind_path(str(target_path))

    def _rehydrate_sub_agents(
        self,
        snapshots: Dict[str, Any],
        histories: Optional[Dict[str, List]] = None,
    ) -> Dict[str, Any]:
        return self.planner.tools_handler.restore_from_snapshot(snapshots, agent_histories=histories)

    def _collect_agent_histories(self) -> Dict[str, List]:
        histories = {
            "planner": [dumpd(m) for m in self.planner.chat_history],
        }
        histories.update(self.planner.tools_handler.collect_subagent_histories())
        return histories

    def _collect_agent_snapshots(self) -> Dict[str, Dict[str, Any]]:
        snapshots = {
            "planner": {
                **self.planner.export_runtime_snapshot(),
            }
        }
        snapshots.update(self.planner.tools_handler.collect_subagent_snapshots())
        return snapshots

    def _merged_events_log(self) -> List[Dict[str, Any]]:
        try:
            from .web_server import session_state as web_session_state

            if getattr(web_session_state, "active_session", None) is self:
                web_events = list(web_session_state.events_log)
                if web_events:
                    return web_events
        except Exception:
            pass

        merged = list(self.local_events_log)
        seen = set()
        unique: List[Dict[str, Any]] = []
        for ev in merged:
            key = (ev.get("timestamp"), ev.get("type"), str(ev.get("data")))
            if key in seen:
                continue
            seen.add(key)
            unique.append(ev)
        return unique

    def _is_active_web_session(self) -> bool:
        try:
            from .web_server import session_state as web_session_state

            return getattr(web_session_state, "active_session", None) is self
        except Exception:
            return False

    @staticmethod
    def _attachment_timeline_payload(attachments: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(attachments or []):
            mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip()
            file_name = str(item.get("file_name") or item.get("fileName") or f"image_{idx + 1}").strip() or f"image_{idx + 1}"
            content = str(item.get("content") or "").strip()
            if not mime_type or not content:
                continue
            events.append(
                {
                    "type": "image",
                    "mime_type": mime_type,
                    "file_name": file_name,
                    "data_url": f"data:{mime_type};base64,{content}",
                }
            )
        return events

    def _record_timeline_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        timestamp: Optional[str] = None,
    ) -> None:
        event = {
            "type": event_type,
            "timestamp": timestamp or self._now_iso(),
            "data": payload,
        }
        self.local_events_log.append(event)
        if self._is_active_web_session():
            return
        try:
            self.store.append_timeline_event(
                self.session_id,
                event,
                revision=int(self.state.get("history_revision", 0)),
            )
        except Exception:
            pass

    def _save_checkpoint(self):
        self.state["history_revision"] = int(self.state.get("history_revision", 0)) + 1

        self.store.save_conversation(
            session_id=self.session_id,
            metadata={
                "input_path": self.state.get("input_path") or self.input_path,
                "output_dir": self.output_dir,
            },
            messages=self.conversation_history,
            agent_state=self.state,
            agent_histories=self._collect_agent_histories(),
            agent_snapshots=self._collect_agent_snapshots(),
            events_log=[],
            compaction_stats=self.state.get("compaction_stats"),
        )

    def _save_conversation(self):
        self._save_checkpoint()

    def abort_interrupted_turn(self, reason: str = "Stopped by user.") -> None:
        abort = getattr(self.planner, "abort_interrupted_turn", None)
        if callable(abort):
            abort(reason)
        if self.conversation_history and self.conversation_history[-1].get("role") == "user":
            self.conversation_history.append({"role": "assistant", "content": reason})

    def compact_context(self) -> Dict[str, Any]:
        from .tools.context_compaction import compact_history, estimate_tokens, get_context_policy

        if not self._runtime_lock.acquire(blocking=False):
            raise RuntimeError("Cannot compact context while an agent turn is still running.")
        try:
            total_before = 0
            total_after = 0
            removed = 0
            any_compacted = False
            compacted_infos: List[Dict[str, Any]] = []

            live_agents = [self.planner]
            if getattr(self.planner.tools_handler, "subagent_manager", None) is not None:
                live_agents.extend(
                    list(self.planner.tools_handler.subagent_manager._agents.values())
                )

            policy = get_context_policy(self.state.get("context_policy"))
            self.state["context_policy"] = dict(policy)
            keep_last_turns = int(policy.get("keep_last_turns", 6))

            for agent in live_agents:
                hist = list(getattr(agent, "chat_history", []) or [])
                before = estimate_tokens(hist)
                compacted, info = compact_history(
                    hist,
                    keep_last_turns=keep_last_turns,
                    llm=self.llm,
                    llm_enabled=bool(policy.get("llm_compact_enabled", True)),
                    llm_input_max_chars=int(policy.get("llm_compact_input_max_chars", 24000)),
                    short_history_max_tool_chars=int(policy.get("max_tool_chars", 1500)),
                    preserve_leading_system=False,
                    state=self.state,
                    trigger="manual",
                    force_full=True,
                    microcompact_enabled=bool(policy.get("microcompact_enabled", True)),
                )
                history_updates = list(info.get("history_updates") or [])
                new_history = history_updates if history_updates else hist
                agent.replace_chat_history(new_history, reset_checkpoint=True)
                total_before += int(info.get("before_tokens", before))
                total_after += int(info.get("after_tokens", estimate_tokens(compacted)))
                removed += int(info.get("removed_messages", 0))
                compacted_infos.append(dict(info))
                if history_updates or str(info.get("mode") or "none") != "none":
                    any_compacted = True

            if not any_compacted:
                return {
                    "before_tokens": total_before,
                    "after_tokens": total_after,
                    "removed_messages": removed,
                    "compaction_stats": dict(self.state.get("compaction_stats") or {}),
                    "changed": False,
                }

            cs = dict(self.state.get("compaction_stats") or {})
            cs["count"] = int(cs.get("count", 0)) + 1
            cs["last_at"] = self._now_iso()
            cs["last_before_tokens"] = total_before
            cs["last_after_tokens"] = total_after
            cs["last_reason"] = "manual_api"
            self.state["compaction_stats"] = cs
            self.state["llm_context_usage"] = {
                "budget_tokens": int(total_after),
                "total_tokens": int(total_after),
                "prompt_tokens": int(total_after),
                "source": "estimate_after_manual_compact",
            }

            self._on_agent_event(
                "context_compacted",
                {
                    "agent": "session",
                    "before_tokens": total_before,
                    "after_tokens": total_after,
                    "removed_messages": removed,
                    "reason": "manual_api",
                    "mode": compacted_infos[-1].get("mode", "rule") if compacted_infos else "rule",
                    "summary_text": compacted_infos[-1].get("summary_text", "") if compacted_infos else "",
                    "summary_core": compacted_infos[-1].get("summary_core", "") if compacted_infos else "",
                },
            )
            self._save_checkpoint()
            return {
                "before_tokens": total_before,
                "after_tokens": total_after,
                "removed_messages": removed,
                "compaction_stats": cs,
                "changed": True,
            }
        finally:
            self._runtime_lock.release()

    def run_turn(self, user_input: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        with self._runtime_lock:
            return self._run_turn_locked(user_input, attachments=attachments)

    def _run_turn_locked(self, user_input: str, attachments: Optional[List[Dict[str, Any]]] = None) -> str:
        attachments = list(attachments or [])
        self.store.append_event(
            self.session_id,
            "turn_started",
            {"input": user_input, "attachment_count": len(attachments)},
            revision=int(self.state.get("history_revision", 0)),
        )

        self.conversation_history.append({"role": "user", "content": user_input, "attachments": attachments})
        self.state["conversation_turn"] = int(self.state.get("conversation_turn", 0)) + 1
        if not self._is_active_web_session():
            self._record_timeline_event(
                "user_message",
                {
                    "content": user_input,
                    "attachments": self._attachment_timeline_payload(attachments),
                },
            )

        message_path = pick_path_from_message(user_input, cwd=os.getcwd())
        if message_path:
            self.state["pending_input_path"] = message_path
            self.input_path = message_path

        self.state["user_goal"]["raw_question"] = user_input
        self.display.print_status("Processing...")

        try:
            # Planner is the single user-facing entry point.
            if str(self.state.get("planner_phase") or "working") == "needs_input":
                self._on_agent_event(
                    "planner_need_user_reply",
                    {"need": dict(self.state.get("planner_need") or {}), "user_reply": user_input},
                )
                self.state["planner_phase"] = "working"
                self.state["planner_need"] = {}

            result = self.planner.run(user_instruction=user_input, attachments=attachments)

            response = "I've processed your request. What would you like to know next?"
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    response = strip_phase_tag(msg.content)
                    break

            self.conversation_history.append({"role": "assistant", "content": response})
            if not self._is_active_web_session():
                self._record_timeline_event("turn_complete", {"response": response})
            self._save_conversation()
            return response
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            if not self._is_active_web_session():
                self._record_timeline_event("error", {"message": error_msg})
            self._save_conversation()
            return error_msg
        finally:
            self.store.append_event(
                self.session_id,
                "turn_finished",
                {"message_count": len(self.conversation_history)},
                revision=int(self.state.get("history_revision", 0)),
            )

    def start(self):
        self._print_session_banner()

        initial_question = self.state["user_goal"].get("raw_question", "Analyze this data")
        self.display.print_user_message(f"(Initial Question) {initial_question}")
        self.display.print_status("Starting initial analysis...")

        response = self.run_turn(initial_question)
        self.display.print_agent_thought("Agent Response", response)
        self._interaction_loop()

    def resume(self):
        self._print_session_banner(resumed=True)

        warnings = list(self.resume_diagnostics.get("warnings") or [])
        if warnings:
            warning_text = "\n".join(f"- {item}" for item in warnings)
            self.display.print_error(f"Resume warnings:\n{warning_text}")

        if self.conversation_history:
            last_assistant = next(
                (
                    str(msg.get("content", ""))
                    for msg in reversed(self.conversation_history)
                    if msg.get("role") == "assistant" and str(msg.get("content", "")).strip()
                ),
                "",
            )
            if last_assistant:
                self.display.print_agent_thought("Last Response", last_assistant)
        self._interaction_loop()

    def _print_session_banner(self, resumed: bool = False) -> None:
        self.display.print_welcome()

        if self.display.console:
            self.display.console.print(f"[dim]📂 Data: {self.input_path or '(not set)'}[/]")
            self.display.console.print(f"[dim]📁 Output: {self.output_dir}[/]")
            self.display.console.print(f"[dim]🧵 Session: {self.session_id}[/]")
            if resumed:
                self.display.console.print("[dim]↩ Resumed existing session[/]")
        else:
            print(f"📂 Data: {self.input_path or '(not set)'}")
            print(f"📁 Output: {self.output_dir}")
            print(f"🧵 Session: {self.session_id}")
            if resumed:
                print("↩ Resumed existing session")

    def _interaction_loop(self):
        while True:
            try:
                user_input = (
                    self.display.console.input("\n[user.name]You > [/]")
                    if self.display.console
                    else input("\nYou > ")
                )
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    if self.display.console:
                        self.display.console.print("\n[bold cyan]👋 Session ended. Thank you for using CytoBridge![/]")
                    else:
                        print("\n👋 Session ended. Thank you for using CytoBridge!")
                    break

                if user_input.lower() == "help":
                    self._print_help()
                    continue

                response = self.run_turn(user_input)
                self.display.print_agent_thought("Agent Response", response)
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except EOFError:
                print("\n👋 Session ended.")
                break

    def _print_help(self):
        help_text = """
Available Commands:
  exit, quit, q  - End the session
  help           - Show this help message

You can ask questions like:
  - "How many cells are in the dataset?"
  - "What model should I use?"
  - "Run downstream trajectory and driver analysis"
"""
        if self.display.console:
            self.display.console.print(help_text)
        else:
            print(help_text)


def run_interactive(
    input_path: Optional[str],
    user_goal: Dict[str, Any],
    openai_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: str = "gpt-4o",
    llm_auth_mode: str = "auto",
    llm_provider: str = "auto",
    llm_profile_id: Optional[str] = None,
    llm_thinking_level: Optional[str] = None,
    output_dir: Optional[str] = None,
    resume_session_id: Optional[str] = None,
) -> None:
    """CLI entrypoint for interactive mode."""
    llm, _ = instantiate_llm(
        model=llm_model,
        base_url=llm_base_url,
        api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        auth_mode=llm_auth_mode,
        provider=llm_provider,
        preferred_profile_id=llm_profile_id,
        thinking_level=llm_thinking_level,
    )
    if resume_session_id:
        session = InteractiveSession.from_checkpoint(resume_session_id, llm)
        session.resume()
        return

    session = InteractiveSession(
        input_path=input_path,
        user_goal=user_goal,
        llm=llm,
        output_dir=output_dir,
    )
    session.start()
