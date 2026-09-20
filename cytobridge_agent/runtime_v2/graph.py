from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .tool_node import build_media_aware_tool_node
from ..tools.turn_phase import PHASE_COMMENTARY, PHASE_FINAL_ANSWER, PHASE_NEEDS_INPUT, resolve_message_phase, should_continue_turn


class RuntimeGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    commentary_retries: int
    stop_hook_triggers: int
    turn_system_prompt: str
    enable_multimodal: bool


def build_runtime_graph(
    agent_node: Callable,
    tools: list,
    post_tool_hook_node: Optional[Callable] = None,
    stop_hook_node: Optional[Callable] = None,
    checkpointer: Optional[Any] = None,
) -> Any:
    workflow = StateGraph(RuntimeGraphState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", build_media_aware_tool_node(tools))
    if post_tool_hook_node is not None:
        workflow.add_node("post_tool_hooks", post_tool_hook_node)
    if stop_hook_node is not None:
        workflow.add_node("stop_hooks", stop_hook_node)
    workflow.set_entry_point("agent")

    def should_continue(state: RuntimeGraphState):
        messages = state["messages"]
        if not messages:
            return END
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            return "tools"
        retries = int(state.get("commentary_retries", 0))
        if should_continue_turn(last_message) and retries < 2:
            return "agent"
        phase = resolve_message_phase(last_message)
        if stop_hook_node is not None and phase in {PHASE_NEEDS_INPUT, PHASE_FINAL_ANSWER}:
            return "stop_hooks"
        if stop_hook_node is not None and phase == PHASE_COMMENTARY:
            return "stop_hooks"
        return END

    def should_continue_after_post_tool_hooks(state: RuntimeGraphState):
        messages = state["messages"]
        if not messages:
            return "agent"
        last_message = messages[-1]
        phase = resolve_message_phase(last_message)
        if stop_hook_node is not None and phase in {PHASE_NEEDS_INPUT, PHASE_FINAL_ANSWER}:
            return "stop_hooks"
        if stop_hook_node is not None and phase == PHASE_COMMENTARY and not should_continue_turn(last_message):
            return "stop_hooks"
        if phase in {PHASE_NEEDS_INPUT, PHASE_FINAL_ANSWER}:
            return END
        return "agent"

    def should_continue_after_stop_hooks(state: RuntimeGraphState):
        messages = state["messages"]
        if not messages:
            return END
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            return "tools"
        retries = int(state.get("commentary_retries", 0))
        if should_continue_turn(last_message) and retries < 2:
            return "agent"
        return END

    agent_edges = {"tools": "tools", "agent": "agent", END: END}
    if stop_hook_node is not None:
        agent_edges["stop_hooks"] = "stop_hooks"
    workflow.add_conditional_edges("agent", should_continue, agent_edges)
    if post_tool_hook_node is not None:
        workflow.add_edge("tools", "post_tool_hooks")
        post_tool_edges = {"agent": "agent", END: END}
        if stop_hook_node is not None:
            post_tool_edges["stop_hooks"] = "stop_hooks"
        workflow.add_conditional_edges(
            "post_tool_hooks",
            should_continue_after_post_tool_hooks,
            post_tool_edges,
        )
    else:
        workflow.add_edge("tools", "agent")
    if stop_hook_node is not None:
        workflow.add_conditional_edges(
            "stop_hooks",
            should_continue_after_stop_hooks,
            {"tools": "tools", "agent": "agent", END: END},
        )
    return workflow.compile(checkpointer=checkpointer)
