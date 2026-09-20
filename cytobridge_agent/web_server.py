
import asyncio
import ctypes
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .display import DisplayManager
from .interactive import InteractiveSession
from .runtime_events import coerce_runtime_event, runtime_event
from .session_controller import SessionController
from .schemas import UserGoal
from .tools.training_algorithm_registry import (
    default_training_algorithm_roots,
    list_training_algorithms,
)
from .tools.training_isolation import terminate_active_training_subprocesses
from .tools.research_idea_registry import (
    list_research_ideas as list_research_idea_catalog,
    load_research_idea_record,
)
from .utils.config_manager import get_saved_config, save_config
from .utils.codex_sidecar_client import get_codex_status_safe, get_codex_usage_safe
from .utils.llm_factory import (
    instantiate_llm,
    normalize_auth_mode,
    normalize_base_url_for_model,
    normalize_llm_reasoning_effort,
)
from .utils.llm_providers import (
    get_llm_provider,
    normalize_llm_provider,
    provider_options,
    resolve_provider_context_window,
)
from .runtime_v2.state import (
    DEFAULT_STOP_HOOK_ENABLED,
    DEFAULT_STOP_HOOK_MAX_TRIGGERS,
    DEFAULT_STOP_HOOK_MODE,
    DEFAULT_STOP_HOOK_PROMPT,
)

# Setup Logger
logger = logging.getLogger("cytobridge_web")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def _web_lifespan(_app: FastAPI):
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()


app = FastAPI(title="CellCompass Agent Web API", lifespan=_web_lifespan)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
class SessionState:
    def __init__(self):
        self.active_session: Optional[InteractiveSession] = None
        self.websockets: list[WebSocket] = []
        self.lock = threading.Lock()
        self.events_log: list[Dict[str, Any]] = []  # Store events for resume
        self.initial_config: Dict[str, Any] = {}
        self.stop_requested: bool = False  # Stop flag for cancelling agent execution
        self.agent_thread: Optional[threading.Thread] = None  # Reference to running agent thread
        self.active_turn_id: Optional[str] = None
        self.recent_chat_requests: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.controller: Optional[SessionController] = None

session_state = SessionState()

WEBSOCKET_SEND_TIMEOUT_SECONDS = 1.0


class AgentStopRequested(BaseException):
    """Raised inside the worker thread when the Web UI requests a hard stop."""


def _raise_in_thread(thread: Optional[threading.Thread], exc_type: type[BaseException]) -> bool:
    """Best-effort Python-level interruption for a worker thread.

    CPython can inject an exception into a thread at the next bytecode checkpoint.
    This is intentionally used only after setting the cooperative stop flag, so
    ordinary tool code that polls stop_check can exit cleanly first.
    """
    if thread is None or not thread.is_alive() or thread.ident is None:
        return False
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(int(thread.ident)),
        ctypes.py_object(exc_type),
    )
    if result == 0:
        return False
    if result > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(int(thread.ident)), None)
        return False
    return True


def _normalize_timeline_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every UI timeline event has a stable envelope."""
    return coerce_runtime_event(event_data, source="web")


def _record_session_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Store a timeline event in-memory for the live UI and durably for resume."""
    event_data = _normalize_timeline_event(event_data)
    session_state.events_log.append(event_data)
    controller = session_state.controller
    session = session_state.active_session or (controller.active_session if controller is not None else None)
    if not session:
        return event_data
    try:
        session.store.append_timeline_event(
            session.session_id,
            event_data,
            revision=int(session.state.get("history_revision", 0)),
        )
    except Exception:
        logger.debug("Failed to persist timeline event", exc_info=True)
    return event_data


CHAT_REQUEST_DEDUPE_TTL_SECONDS = 10 * 60
CHAT_REQUEST_DEDUPE_MAX = 256


def _normalize_client_request_id(value: Optional[str]) -> str:
    text = str(value or "").strip()
    return text[:160]


def _prune_recent_chat_requests(now: Optional[float] = None) -> None:
    now = float(now if now is not None else time.time())
    while session_state.recent_chat_requests:
        first_key = next(iter(session_state.recent_chat_requests))
        first = session_state.recent_chat_requests.get(first_key) or {}
        if now - float(first.get("created_at") or 0) <= CHAT_REQUEST_DEDUPE_TTL_SECONDS:
            break
        session_state.recent_chat_requests.popitem(last=False)
    while len(session_state.recent_chat_requests) > CHAT_REQUEST_DEDUPE_MAX:
        session_state.recent_chat_requests.popitem(last=False)


def _duplicate_chat_response(client_request_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "success",
        "message": "Duplicate chat request ignored",
        "duplicate": True,
        "client_request_id": client_request_id,
        "turn_id": record.get("turn_id"),
        "request_status": record.get("status") or "accepted",
    }


def _reserve_chat_request(client_request_id: str, turn_id: str) -> Optional[Dict[str, Any]]:
    if not client_request_id:
        return None
    now = time.time()
    with session_state.lock:
        _prune_recent_chat_requests(now)
        existing = session_state.recent_chat_requests.get(client_request_id)
        if existing:
            return dict(existing)
        session_state.recent_chat_requests[client_request_id] = {
            "turn_id": turn_id,
            "status": "accepted",
            "created_at": now,
        }
    return None


def _mark_chat_request_status(client_request_id: str, status: str) -> None:
    if not client_request_id:
        return
    with session_state.lock:
        record = session_state.recent_chat_requests.get(client_request_id)
        if record is not None:
            record["status"] = status
            record["updated_at"] = time.time()


def _forget_chat_request(client_request_id: str) -> None:
    if not client_request_id:
        return
    with session_state.lock:
        session_state.recent_chat_requests.pop(client_request_id, None)

# --- Pydantic Models ---
class InitRequest(BaseModel):
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_provider: str = "auto"
    llm_auth_mode: str = "auto"
    llm_profile_id: Optional[str] = None
    llm_thinking_level: str = "low"
    llm_model: str = "gpt-4"
    device: str = "cuda"
    enable_multimodal: bool = True  # Multi-modal toggle
    tool_activation_mode: str = "auto_next_turn"
    tool_harvest_enabled: bool = False
    algorithm_proposal_review_mode: str = "agent_decide"
    idea_review_mode: str = "agent_decide"
    stop_hook_enabled: bool = DEFAULT_STOP_HOOK_ENABLED
    stop_hook_mode: str = DEFAULT_STOP_HOOK_MODE
    stop_hook_prompt: str = DEFAULT_STOP_HOOK_PROMPT
    stop_hook_max_triggers: int = DEFAULT_STOP_HOOK_MAX_TRIGGERS
    allow_large_context_window: bool = False


class ModelSwitchRequest(BaseModel):
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_auth_mode: Optional[str] = None
    llm_profile_id: Optional[str] = None
    llm_thinking_level: Optional[str] = None
    algorithm_proposal_review_mode: Optional[str] = None
    idea_review_mode: Optional[str] = None
    stop_hook_enabled: Optional[bool] = None
    stop_hook_mode: Optional[str] = None
    stop_hook_prompt: Optional[str] = None
    stop_hook_max_triggers: Optional[int] = None
    allow_large_context_window: Optional[bool] = None
    persist: bool = True


def _request_field_was_set(request: Optional[BaseModel], field_name: str) -> bool:
    """Return whether a Pydantic request explicitly provided a field."""

    if request is None:
        return False
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(request, "__fields_set__", set())
    return field_name in fields_set


def _normalize_base_url_for_model(model: str, base_url: Optional[str]) -> Optional[str]:
    value = (base_url or "").strip()
    if not model.startswith("gemini"):
        return value or None
    # Treat default OpenAI URL as "unset" for Gemini OAuth mode.
    if value in {"", "https://api.openai.com/v1", "https://api.openai.com/v1/"}:
        return None
    return value

def _normalize_algorithm_proposal_review_mode(value: Optional[str]) -> str:
    mode = str(value or "agent_decide").strip().lower()
    if mode in {"always", "always_user_review", "manual"}:
        return "always_user_review"
    if mode in {"agent", "agent_decide", "on_uncertainty"}:
        return "agent_decide"
    if mode in {"auto", "auto_approve", "always_auto"}:
        return "auto_approve"
    return "agent_decide"


def _normalize_idea_review_mode(value: Optional[str]) -> str:
    mode = str(value or "agent_decide").strip().lower()
    if mode in {"always", "always_user_review", "manual"}:
        return "always_user_review"
    if mode in {"agent", "agent_decide", "on_uncertainty"}:
        return "agent_decide"
    if mode in {"auto", "auto_approve", "always_auto"}:
        return "auto_approve"
    return "agent_decide"


def _normalize_stop_hook_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(value)


def _normalize_stop_hook_mode(value: Optional[Any]) -> str:
    mode = str(value or DEFAULT_STOP_HOOK_MODE).strip().lower()
    if mode in {"prompt", "llm"}:
        return "prompt"
    if mode in {"agent", "subagent"}:
        return "agent"
    return DEFAULT_STOP_HOOK_MODE


def _normalize_stop_hook_prompt(value: Optional[Any]) -> str:
    return str(value or DEFAULT_STOP_HOOK_PROMPT).strip()


def _normalize_stop_hook_max_triggers(value: Optional[Any], default: int = DEFAULT_STOP_HOOK_MAX_TRIGGERS) -> int:
    try:
        normalized = int(value if value is not None else default)
    except Exception:
        normalized = int(default)
    return max(1, min(normalized, 50))


def _json_response_safe(value: Any) -> Any:
    """Convert API payloads to strict JSON values accepted by Starlette."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_response_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_response_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _json_response_safe(value.tolist())
        if isinstance(value, np.generic):
            return _json_response_safe(value.item())
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value, allow_nan=False)
        return value
    except Exception:
        return str(value)


def _slash_event(message: str, *, level: str = "info", command: str = "", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "message": message,
        "level": level,
        "command": command,
    }
    if data:
        payload.update(data)
    return _normalize_timeline_event(
        {
            "type": "system",
            "timestamp": datetime.now().isoformat(),
            "data": payload,
        }
    )


def _record_slash_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recorded: List[Dict[str, Any]] = []
    for event in events:
        recorded.append(_record_session_event(event))
    return recorded


def _slash_success(message: str, *, command: str = "", events: Optional[List[Dict[str, Any]]] = None, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    all_events = list(events or [])
    all_events.append(_slash_event(message, command=command, data=data))
    return {
        "status": "success",
        "command_handled": True,
        "message": message,
        "events": _record_slash_events(all_events),
    }


def _slash_error(message: str, *, command: str = "") -> Dict[str, Any]:
    event = _slash_event(message, level="error", command=command)
    return {
        "status": "error",
        "command_handled": True,
        "message": message,
        "events": _record_slash_events([event]),
    }


def _slash_help_message() -> str:
    return "\n".join(
        [
            "Slash commands:",
            "/help - show this help",
            "/new - clear the active in-memory session",
            "/stop - request the current agent turn to stop",
            "/resume SESSION_ID - resume a saved session",
            "/status - show active session/model/hook status",
            "/reasoning [off|low|medium|high|xhigh|max] - show or update reasoning/thinking level",
            "/model [model_id] - show or switch model within the current provider",
            "/provider [provider_id] [model_id] - show or switch provider, optionally with model",
            "/compact - manually compact the active session context",
            "/hooks [status|on|off|max N|prompt TEXT] - inspect or update stop hook settings",
            "/skills [planner|workflow|downstream] - list exposed skills for a scope",
        ]
    )


def _resolve_llm_runtime_config(request: Optional[BaseModel] = None) -> Dict[str, Optional[str]]:
    saved_config = get_saved_config()
    cli_config = session_state.initial_config or {}

    req_model = getattr(request, "llm_model", None) if request is not None else None
    req_api_key = getattr(request, "openai_api_key", None) if request is not None else None
    req_provider = getattr(request, "llm_provider", None) if request is not None else None
    req_auth_mode = getattr(request, "llm_auth_mode", None) if request is not None else None
    req_profile_id = getattr(request, "llm_profile_id", None) if request is not None else None
    req_thinking_level = getattr(request, "llm_thinking_level", None) if request is not None else None
    req_has_provider = _request_field_was_set(request, "llm_provider")
    req_has_profile = _request_field_was_set(request, "llm_profile_id")
    req_has_thinking = _request_field_was_set(request, "llm_thinking_level")
    req_has_large_context = _request_field_was_set(request, "allow_large_context_window")
    req_has_base = request is not None and getattr(request, "llm_base_url", None) is not None
    req_base_url = getattr(request, "llm_base_url", None) if request is not None else None

    model = (
        (req_model.strip() if isinstance(req_model, str) and req_model.strip() else None)
        or cli_config.get("llm_model")
        or saved_config.get("llm_model")
        or "gpt-4o"
    )
    provider = normalize_llm_provider(
        (req_provider if req_provider is not None else None)
        or cli_config.get("llm_provider")
        or saved_config.get("llm_provider")
        or "auto"
    )
    req_api_key_value = req_api_key.strip() if isinstance(req_api_key, str) and req_api_key.strip() else None
    cli_provider = normalize_llm_provider(cli_config.get("llm_provider") or "auto")
    saved_provider = normalize_llm_provider(saved_config.get("llm_provider") or "auto")
    provider_api_keys = _merged_provider_api_keys(saved_config, cli_config)
    provider_api_key = provider_api_keys.get(provider)
    provider_base_urls = _merged_provider_base_urls(saved_config, cli_config)
    provider_base_url = provider_base_urls.get(provider)
    cli_api_key = cli_config.get("openai_api_key") if provider in {"auto", cli_provider} else None
    saved_api_key = saved_config.get("openai_api_key") if provider in {"auto", saved_provider} else None
    api_key = (
        req_api_key_value
        or provider_api_key
        or cli_api_key
        or saved_api_key
        or (os.environ.get("OPENAI_API_KEY") if provider == "auto" else None)
    )

    if req_has_base:
        base_url = req_base_url
    else:
        # Provider-specific base URLs must not leak across provider switches.
        # Example: switching from Xiaomi to DeepSeek without specifying
        # llm_base_url should use DeepSeek's provider default, not Xiaomi's
        # token-plan endpoint from the previous runtime config.
        base_url = provider_base_url
        if base_url is None and provider in {"auto", cli_provider}:
            base_url = cli_config.get("llm_base_url")
        if base_url is None and provider in {"auto", saved_provider}:
            base_url = saved_config.get("llm_base_url")
    auth_mode = normalize_auth_mode(
        (req_auth_mode if req_auth_mode is not None else None)
        or cli_config.get("llm_auth_mode")
        or saved_config.get("llm_auth_mode")
        or "auto"
    )
    if req_has_profile:
        profile_id = req_profile_id.strip() if isinstance(req_profile_id, str) and req_profile_id.strip() else None
    else:
        profile_id = cli_config.get("llm_profile_id") or saved_config.get("llm_profile_id")
    if req_has_thinking:
        thinking_level = normalize_llm_reasoning_effort(req_thinking_level, default="low")
    elif req_has_provider and provider == "xiaomi":
        # Xiaomi thinking-mode histories require strict reasoning_content
        # round-tripping. Provider fallback should not inherit a prior thinking
        # level unless the caller explicitly asks for it.
        thinking_level = "off"
    else:
        thinking_level = normalize_llm_reasoning_effort(
            cli_config.get("llm_thinking_level") or saved_config.get("llm_thinking_level"),
            default="low",
        )
    if req_has_large_context:
        allow_large_context_window = bool(getattr(request, "allow_large_context_window", False))
    else:
        allow_large_context_window = bool(
            cli_config.get("allow_large_context_window", saved_config.get("allow_large_context_window", False))
        )

    if auth_mode == "gemini_oauth":
        base_url = None
    elif auth_mode == "auto":
        base_url = _normalize_base_url_for_model(model, base_url)
    else:
        base_url = (base_url or "").strip() or None

    return {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "auth_mode": auth_mode,
        "profile_id": profile_id,
        "thinking_level": thinking_level,
        "allow_large_context_window": allow_large_context_window,
    }


def _instantiate_runtime_llm(config: Dict[str, Optional[str]]):
    return instantiate_llm(
        model=config.get("model"),
        base_url=config.get("base_url"),
        api_key=config.get("api_key"),
        auth_mode=config.get("auth_mode"),
        provider=config.get("provider"),
        preferred_profile_id=config.get("profile_id"),
        thinking_level=config.get("thinking_level"),
    )


def _controller_config_from_runtime(runtime_cfg: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {
        "llm_model": runtime_cfg.get("model"),
        "llm_provider": runtime_cfg.get("provider"),
        "llm_base_url": runtime_cfg.get("base_url"),
        "llm_api_key": runtime_cfg.get("api_key"),
        "llm_auth_mode": runtime_cfg.get("auth_mode"),
        "llm_profile_id": runtime_cfg.get("profile_id"),
        "llm_thinking_level": runtime_cfg.get("thinking_level"),
        "allow_large_context_window": runtime_cfg.get("allow_large_context_window"),
    }


def _web_controller(runtime_cfg: Optional[Dict[str, Optional[str]]] = None) -> SessionController:
    cfg = dict(_controller_config_from_runtime(runtime_cfg or _resolve_llm_runtime_config()))

    def llm_factory(overrides: Optional[Dict[str, Any]] = None):
        merged = dict(cfg)
        merged.update({k: v for k, v in dict(overrides or {}).items() if v is not None})
        llm, _ = instantiate_llm(
            model=merged.get("llm_model"),
            base_url=merged.get("llm_base_url"),
            api_key=merged.get("llm_api_key"),
            auth_mode=merged.get("llm_auth_mode"),
            provider=merged.get("llm_provider"),
            preferred_profile_id=merged.get("llm_profile_id"),
            thinking_level=merged.get("llm_thinking_level"),
        )
        return llm

    controller = session_state.controller
    if controller is None:
        controller = SessionController(llm_factory, initial_config=cfg)
        session_state.controller = controller
    else:
        controller.llm_factory = llm_factory
        controller.initial_config.update(cfg)
    return controller


def _adopt_web_session(session: InteractiveSession, runtime_cfg: Dict[str, Optional[str]]) -> None:
    controller = _web_controller(runtime_cfg)

    def check_stop() -> bool:
        return session_state.stop_requested

    controller.adopt_session(session, stop_check=check_stop, owner_kind="web")


def _runtime_context_window(config: Dict[str, Optional[str]]) -> int:
    return resolve_provider_context_window(
        config.get("provider"),
        config.get("model"),
    )


def _sync_session_context_window(session: InteractiveSession, config: Dict[str, Optional[str]]) -> int:
    from .tools.context_compaction import get_context_policy

    window = _runtime_context_window(config)
    policy = get_context_policy(session.state.get("context_policy"))
    policy["context_window"] = int(window)
    policy["allow_large_context_window"] = bool(config.get("allow_large_context_window", False))
    policy = get_context_policy(policy)
    session.state["context_policy"] = policy
    return int(window)


def _active_runtime_status() -> Dict[str, Any]:
    cfg = _resolve_llm_runtime_config()
    controller = session_state.controller or _web_controller(cfg)
    status = dict(controller.status())
    session = session_state.active_session or controller.active_session
    state = getattr(session, "state", {}) if session else {}
    status.update({
        "has_active_session": bool(session),
        "is_agent_running": bool(session_state.agent_thread and session_state.agent_thread.is_alive()),
        "session_id": getattr(session, "session_id", None) if session else None,
        "model": cfg.get("model"),
        "provider": cfg.get("provider"),
        "auth_mode": cfg.get("auth_mode"),
        "thinking_level": cfg.get("thinking_level"),
        "context_window": _runtime_context_window(cfg),
        "context_policy": dict(state.get("context_policy") or {}),
        "llm_context_usage": dict(state.get("llm_context_usage") or {}),
        "stop_hook_enabled": _normalize_stop_hook_enabled(
            state.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
        ),
        "stop_hook_mode": _normalize_stop_hook_mode(
            state.get("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
        ),
        "stop_hook_max_triggers": _normalize_stop_hook_max_triggers(
            state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS)
        ),
    })
    snapshot = status.get("snapshot")
    if isinstance(snapshot, dict):
        snapshot = dict(snapshot)
        snapshot["adapter"] = {
            "name": "web",
            "is_agent_running": status["is_agent_running"],
            "context_window": status["context_window"],
            "auth_mode": status["auth_mode"],
        }
        status["snapshot"] = snapshot
    return status


def _format_status_message(status: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Session status:",
            f"active: {status.get('has_active_session')}",
            f"running: {status.get('is_agent_running')}",
            f"session_id: {status.get('session_id') or 'none'}",
            f"provider: {status.get('provider') or 'auto'}",
            f"model: {status.get('model') or 'unknown'}",
            f"reasoning: {status.get('thinking_level') or 'low'}",
            f"context_window: {status.get('context_window') or 'unknown'}",
            f"compact_effective_window: {(status.get('context_policy') or {}).get('effective_context_window') or 'unknown'}",
            (
                "stop_hook: "
                f"{'on' if status.get('stop_hook_enabled') else 'off'} "
                f"mode={status.get('stop_hook_mode')} "
                f"max={status.get('stop_hook_max_triggers')}"
            ),
        ]
    )


def _update_stop_hook_settings(args: List[str], rest: str) -> Dict[str, Any]:
    if not session_state.active_session:
        raise RuntimeError("Session not initialized")
    state = session_state.active_session.state
    if not args or args[0].lower() == "status":
        return {}

    updates: Dict[str, Any] = {}
    first = args[0].strip().lower()
    if first in {"on", "enable", "enabled", "true", "1"}:
        updates["stop_hook_enabled"] = True
        args = args[1:]
    elif first in {"off", "disable", "disabled", "false", "0"}:
        updates["stop_hook_enabled"] = False
        args = args[1:]
    elif first in {"max", "limit"} and len(args) >= 2:
        updates["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(args[1])
        args = args[2:]
    elif first.startswith("max="):
        updates["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(first.split("=", 1)[1])
        args = args[1:]
    elif first in {"prompt", "message"}:
        split_rest = rest.split(None, 1)
        prompt_text = split_rest[1].strip() if len(split_rest) > 1 else ""
        updates["stop_hook_mode"] = "prompt"
        updates["stop_hook_prompt"] = _normalize_stop_hook_prompt(prompt_text)
        args = []

    idx = 0
    while idx < len(args):
        token = args[idx].strip().lower()
        if token in {"max", "limit"} and idx + 1 < len(args):
            updates["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(args[idx + 1])
            idx += 2
            continue
        if token.startswith("max="):
            updates["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(token.split("=", 1)[1])
        idx += 1

    for key, value in updates.items():
        state[key] = value
        session_state.initial_config[key] = value
    if "stop_hook_enabled" in updates:
        try:
            session_state.active_session.planner.refresh_tooling()
        except Exception:
            logger.debug("Failed to refresh tooling after stop-hook enablement update", exc_info=True)
    if updates:
        save_config(updates)
    return updates


class ChatRequest(BaseModel):
    message: str
    attachments: Optional[List[Dict[str, Any]]] = None
    client_request_id: Optional[str] = None


class StopHookSettingsRequest(BaseModel):
    stop_hook_enabled: Optional[bool] = None
    stop_hook_mode: Optional[str] = None
    stop_hook_prompt: Optional[str] = None
    stop_hook_max_triggers: Optional[int] = None


class ToolActivationModeRequest(BaseModel):
    mode: str


class ToolHarvestModeRequest(BaseModel):
    enabled: bool


class ApproveToolsRequest(BaseModel):
    tool_ids: Optional[List[str]] = None


class DisableToolRequest(BaseModel):
    name_or_id: str


class SkillMutationRequest(BaseModel):
    scope: str = "planner"
    name: str


class SkillExposureRequest(BaseModel):
    scope: str = "planner"
    name: str
    exposed: bool = True


class AlgorithmExposureRequest(BaseModel):
    name: str
    exposed: bool = True


class ProposalReviewRequest(BaseModel):
    algorithm_id: str
    proposal_id: str = ""
    decision: str
    reviewer_feedback: str = ""
    auto_continue: bool = True


class ResearchIdeaReviewRequest(BaseModel):
    idea_id: str
    decision: str
    reviewer_feedback: str = ""
    auto_continue: bool = True


class ActiveResearchIdeaRequest(BaseModel):
    idea_id: str


def _is_slash_command(message: str) -> bool:
    return str(message or "").strip().startswith("/")


def _parse_slash_command(message: str) -> tuple[str, List[str], str]:
    text = str(message or "").strip()
    if not text.startswith("/"):
        return "", [], ""
    command_text = text[1:].strip()
    if not command_text:
        return "help", [], ""
    parts = command_text.split()
    name = parts[0].strip().lower()
    args = parts[1:]
    rest = command_text[len(parts[0]):].strip() if parts else ""
    aliases = {
        "?": "help",
        "commands": "help",
        "command": "help",
        "think": "reasoning",
        "thinking": "reasoning",
        "reason": "reasoning",
        "hook": "hooks",
        "stop_hook": "hooks",
        "stop-hooks": "hooks",
        "providers": "provider",
        "skill": "skills",
    }
    return aliases.get(name, name), args, rest


def _normalize_skill_scope(scope: Optional[str]) -> str:
    selected = (scope or "planner").strip().lower()
    if selected not in {"workflow", "planner", "downstream"}:
        raise ValueError("Invalid scope. Use 'workflow', 'planner', or 'downstream'.")
    return selected


def _normalize_name_list(items: Any) -> List[str]:
    out: List[str] = []
    for item in (items or []):
        name = str(item or "").strip().lower()
        if name and name not in out:
            out.append(name)
    return out


def _normalize_hidden_skills(raw: Any) -> Dict[str, List[str]]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "workflow": _normalize_name_list(data.get("workflow", [])),
        "planner": _normalize_name_list(data.get("planner", [])),
        "downstream": _normalize_name_list(data.get("downstream", [])),
    }


def _normalize_hidden_training_algorithms(raw: Any) -> List[str]:
    return _normalize_name_list(raw)


def _normalize_provider_api_keys(raw: Any) -> Dict[str, str]:
    data = raw if isinstance(raw, dict) else {}
    out: Dict[str, str] = {}
    for key, value in data.items():
        provider = normalize_llm_provider(key)
        api_key = str(value or "").strip()
        if provider and api_key:
            out[provider] = api_key
    return out


def _merged_provider_api_keys(*configs: Dict[str, Any]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for config in configs:
        if isinstance(config, dict):
            merged.update(_normalize_provider_api_keys(config.get("provider_api_keys")))
    return merged


def _normalize_provider_base_urls(raw: Any) -> Dict[str, str]:
    data = raw if isinstance(raw, dict) else {}
    out: Dict[str, str] = {}
    for key, value in data.items():
        provider = normalize_llm_provider(key)
        base_url = str(value or "").strip()
        if provider and base_url:
            out[provider] = base_url
    return out


def _merged_provider_base_urls(*configs: Dict[str, Any]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for config in configs:
        if isinstance(config, dict):
            merged.update(_normalize_provider_base_urls(config.get("provider_base_urls")))
    return merged


def _provider_key_save_payload(provider: str, api_key: Optional[str]) -> Dict[str, Any]:
    normalized_provider = normalize_llm_provider(provider)
    explicit_key = str(api_key or "").strip()
    payload: Dict[str, Any] = {}

    saved = get_saved_config()
    existing_keys = _merged_provider_api_keys(saved, session_state.initial_config)

    legacy_key = str(saved.get("openai_api_key") or session_state.initial_config.get("openai_api_key") or "").strip()
    legacy_provider = normalize_llm_provider(
        session_state.initial_config.get("llm_provider")
        or saved.get("llm_provider")
        or "auto"
    )
    if legacy_key and legacy_provider not in {"auto", "openai-compatible"}:
        existing_keys.setdefault(legacy_provider, legacy_key)

    if explicit_key and normalized_provider not in {"auto", "openai-compatible"}:
        existing_keys[normalized_provider] = explicit_key
        payload["provider_api_keys"] = existing_keys
    elif existing_keys:
        payload["provider_api_keys"] = existing_keys

    if explicit_key and normalized_provider in {"auto", "openai-compatible", "openai", "openrouter"}:
        payload["openai_api_key"] = explicit_key
        if normalized_provider not in {"auto", "openai-compatible"}:
            existing_keys[normalized_provider] = explicit_key
            payload["provider_api_keys"] = existing_keys
    return payload


def _provider_base_url_save_payload(provider: str, base_url: Optional[str]) -> Dict[str, Any]:
    normalized_provider = normalize_llm_provider(provider)
    explicit_base_url = str(base_url or "").strip()
    if not explicit_base_url or normalized_provider in {"auto", "openai-compatible"}:
        return {}

    saved = get_saved_config()
    existing_base_urls = _merged_provider_base_urls(saved, session_state.initial_config)
    existing_base_urls[normalized_provider] = explicit_base_url
    return {"provider_base_urls": existing_base_urls}


def _resolve_exposure_settings() -> Dict[str, Any]:
    saved = get_saved_config()
    initial = session_state.initial_config or {}
    hidden_skills = _normalize_hidden_skills(
        initial.get("hidden_skills", saved.get("hidden_skills", {}))
    )
    hidden_training_algorithms = _normalize_hidden_training_algorithms(
        initial.get("hidden_training_algorithms", saved.get("hidden_training_algorithms", []))
    )
    return {
        "hidden_skills": hidden_skills,
        "hidden_training_algorithms": hidden_training_algorithms,
    }


def _apply_exposure_settings_to_state(state: Dict[str, Any]) -> Dict[str, Any]:
    settings = _resolve_exposure_settings()
    state["hidden_skills"] = _normalize_hidden_skills(settings.get("hidden_skills"))
    state["hidden_training_algorithms"] = _normalize_hidden_training_algorithms(
        settings.get("hidden_training_algorithms")
    )
    return settings


def _persist_exposure_settings(hidden_skills: Dict[str, List[str]], hidden_training_algorithms: List[str]) -> None:
    payload = {
        "hidden_skills": _normalize_hidden_skills(hidden_skills),
        "hidden_training_algorithms": _normalize_hidden_training_algorithms(hidden_training_algorithms),
    }
    save_config(payload)
    session_state.initial_config.update(payload)


def _list_training_algorithms_with_visibility(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    hidden = {
        str(item or "").strip().lower()
        for item in (state.get("hidden_training_algorithms") or [])
        if str(item or "").strip()
    }
    records = list_training_algorithms(default_training_algorithm_roots())
    items: List[Dict[str, Any]] = []
    for rec in records:
        name = str(rec.algorithm_id or "")
        lowered = name.strip().lower()
        items.append(
            {
                "name": name,
                "description": rec.description or "",
                "requirements": rec.requirements or "",
                "source": rec.source,
                "path": str(rec.root_dir),
                "manifest_path": str(rec.manifest_path),
                "enabled": lowered not in hidden,
            }
        )
    items.sort(key=lambda x: x.get("name", ""))
    return items


def _list_research_ideas_for_ui(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    active_id = str(state.get("active_research_idea_id") or "").strip().lower()
    items: List[Dict[str, Any]] = []
    for item in list_research_idea_catalog():
        normalized = dict(item)
        normalized["active"] = bool(item.get("active")) or (
            active_id and str(item.get("idea_id") or "").strip().lower() == active_id
        )
        items.append(normalized)
    items.sort(
        key=lambda entry: (
            0 if entry.get("active") else 1,
            str(entry.get("updated_at") or ""),
            str(entry.get("idea_id") or ""),
        )
    )
    return items


def _build_detached_skills_tools(scope: str):
    from .tools.skills_tools import SkillsTools

    selected = _normalize_skill_scope(scope)
    visibility = _resolve_exposure_settings()
    if selected == "workflow":
        state = {
            "active_skills": [],
            "skills_revision": 0,
            "skill_policy": {"enabled": True, "domain": "workflow"},
            "hidden_skills": visibility["hidden_skills"],
            "hidden_training_algorithms": visibility["hidden_training_algorithms"],
        }
        tools = SkillsTools(state, domain="workflow")
        return tools, state
    if selected == "planner":
        state = {
            "planner_loaded_skills": [],
            "planner_skills_revision": 0,
            "planner_skill_policy": {
                "enabled": True,
                "domain": "planner",
                "prefer_user_dir": True,
                "inject_max_chars": 24000,
            },
            "planner_active_auto_skills": [],
            "hidden_skills": visibility["hidden_skills"],
            "hidden_training_algorithms": visibility["hidden_training_algorithms"],
        }
        tools = SkillsTools(
            state,
            domain="planner",
            active_key="planner_loaded_skills",
            revision_key="planner_skills_revision",
            policy_key="planner_skill_policy",
        )
        return tools, state

    state = {
        "active_skills": [],
        "skills_revision": 0,
        "skill_policy": {"enabled": True, "domain": "downstream"},
        "hidden_skills": visibility["hidden_skills"],
        "hidden_training_algorithms": visibility["hidden_training_algorithms"],
    }
    tools = SkillsTools(state, domain="downstream")
    return tools, state


def _resolve_session_skills_tools(scope: str):
    selected = _normalize_skill_scope(scope)
    session = session_state.active_session
    if not session:
        return _build_detached_skills_tools(selected)

    state = session.state
    if "hidden_skills" not in state or "hidden_training_algorithms" not in state:
        _apply_exposure_settings_to_state(state)
    planner_tools = session.planner.tools_handler
    if selected == "workflow":
        from .tools.skills_tools import SkillsTools

        tools = SkillsTools(state, domain="workflow", event_sink=planner_tools._emit_event)
        return tools, state
    if selected == "planner":
        return planner_tools.skills_tools, state

    from .tools.skills_tools import SkillsTools

    tools = SkillsTools(state, domain="downstream", event_sink=planner_tools._emit_event)
    return tools, state


async def _handle_slash_command(message: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if not _is_slash_command(message):
        return None
    if attachments:
        return _slash_error("Slash commands do not accept attachments.", command="slash")

    command, args, rest = _parse_slash_command(message)
    try:
        if command in {"help", ""}:
            return _slash_success(_slash_help_message(), command="help")

        if command == "stop":
            result = await stop_agent()
            if result.get("status") != "success":
                return _slash_error(f"Failed to stop agent: {result.get('message')}", command="stop")
            return _slash_success(result.get("message") or "Stop signal sent.", command="stop")

        if command == "new":
            result = await new_session()
            if result.get("status") != "success":
                return _slash_error(f"Failed to start a new session: {result.get('message')}", command="new")
            return {
                "status": "success",
                "command_handled": True,
                "message": result.get("message") or "Active session cleared.",
                "clear_timeline": True,
                "events": [
                    _slash_event(
                        "Active session cleared. Open setup or start a fresh chat.",
                        command="new",
                    )
                ],
            }

        if command == "resume":
            if session_state.agent_thread and session_state.agent_thread.is_alive():
                return _slash_error("Cannot resume while the agent is running. Use /stop first.", command="resume")
            if not args:
                return _slash_error("Usage: /resume SESSION_ID", command="resume")
            session_id = args[0].strip()
            result = await resume_session(session_id)
            if result.get("status") != "success":
                return _slash_error(f"Failed to resume session: {result.get('message')}", command="resume")
            config = await get_config()
            return {
                "status": "success",
                "command_handled": True,
                "message": f"Session resumed: {session_id}",
                "resume": {
                    "session_id": session_id,
                    "config": config,
                    "resume_result": result,
                },
                "events": [
                    _slash_event(
                        f"Session resumed: {session_id}",
                        command="resume",
                        data={"session_id": session_id},
                    )
                ],
            }

        if command == "status":
            status = _active_runtime_status()
            return _slash_success(_format_status_message(status), command="status", data={"status": status})

        if command == "reasoning":
            if not args:
                status = _active_runtime_status()
                return _slash_success(
                    f"Current reasoning/thinking level: {status.get('thinking_level') or 'low'}",
                    command="reasoning",
                    data={"status": status},
                )
            level = normalize_llm_reasoning_effort(args[0], default="low")
            result = await switch_model(ModelSwitchRequest(llm_thinking_level=level, persist=True))
            if result.get("status") != "success":
                return _slash_error(f"Failed to update reasoning: {result.get('message')}", command="reasoning")
            return _slash_success(f"Reasoning/thinking level set to {level}.", command="reasoning")

        if command == "model":
            if not args:
                status = _active_runtime_status()
                return _slash_success(
                    f"Current model: {status.get('model') or 'unknown'}",
                    command="model",
                    data={"status": status},
                )
            model = " ".join(args).strip()
            result = await switch_model(ModelSwitchRequest(llm_model=model, persist=True))
            if result.get("status") != "success":
                return _slash_error(f"Failed to switch model: {result.get('message')}", command="model")
            return _slash_success(f"Model set to {result.get('model') or model}.", command="model")

        if command == "provider":
            if not args:
                status = _active_runtime_status()
                return _slash_success(
                    f"Current provider: {status.get('provider') or 'auto'}",
                    command="provider",
                    data={"status": status},
                )
            provider = args[0]
            model = " ".join(args[1:]).strip() or get_llm_provider(provider).default_model or None
            result = await switch_model(ModelSwitchRequest(llm_provider=provider, llm_model=model, persist=True))
            if result.get("status") != "success":
                return _slash_error(f"Failed to switch provider: {result.get('message')}", command="provider")
            return _slash_success(
                f"Provider set to {normalize_llm_provider(provider)}"
                + (f", model {result.get('model') or model}." if model else "."),
                command="provider",
            )

        if command == "compact":
            if not session_state.active_session:
                return _slash_error("Session not initialized.", command="compact")
            if session_state.agent_thread and session_state.agent_thread.is_alive():
                return _slash_error("Cannot compact while the agent is running.", command="compact")
            compact_info = session_state.active_session.compact_context()
            events: List[Dict[str, Any]] = []
            if compact_info.get("changed"):
                events.append(
                    _normalize_timeline_event(
                        {
                            "type": "context_compacted",
                            "timestamp": datetime.now().isoformat(),
                            "data": {"source": "slash_command", **compact_info},
                        }
                    )
                )
            before = compact_info.get("before_tokens", 0)
            after = compact_info.get("after_tokens", 0)
            return _slash_success(
                f"Compact complete. tokens: {before} -> {after}. changed={bool(compact_info.get('changed'))}",
                command="compact",
                events=events,
                data={"compact_result": compact_info},
            )

        if command == "hooks":
            updates = _update_stop_hook_settings(args, rest)
            status = _active_runtime_status()
            prefix = "Stop hook updated." if updates else "Stop hook status."
            return _slash_success(
                prefix
                + f" enabled={status.get('stop_hook_enabled')} mode={status.get('stop_hook_mode')} max={status.get('stop_hook_max_triggers')}",
                command="hooks",
                data={"status": status, "updates": updates},
            )

        if command == "skills":
            scope = args[0] if args else "planner"
            selected = _normalize_skill_scope(scope)
            skills_tools, _state = _resolve_session_skills_tools(selected)
            available_payload = json.loads(skills_tools.list_skills())
            skills = list(available_payload.get("skills") or [])
            lines = [f"Skills ({selected}): {len(skills)} available."]
            for item in skills[:30]:
                name = str(item.get("name") or "").strip()
                path = str(item.get("skill_md_path") or "").strip()
                desc = str(item.get("description") or "").strip()
                lines.append(f"- {name}: {desc} ({path})")
            if len(skills) > 30:
                lines.append(f"... {len(skills) - 30} more")
            return _slash_success("\n".join(lines), command="skills", data={"scope": selected, "count": len(skills)})

        return _slash_error(f"Unknown slash command: /{command}. Use /help.", command=command)
    except ValueError as exc:
        return _slash_error(str(exc), command=command)
    except Exception as exc:
        logger.exception("Slash command failed: /%s", command)
        return _slash_error(str(exc), command=command)


# --- WebSocket Manager ---
async def broadcast_event(event_data: Dict[str, Any]):
    """Send an event to all connected clients and log for resume."""
    event_data = _record_session_event(event_data)

    disconnected = []
    for ws in list(session_state.websockets):
        try:
            await asyncio.wait_for(
                ws.send_json(event_data),
                timeout=WEBSOCKET_SEND_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Dropping slow websocket client after send timeout")
            disconnected.append(ws)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected
    for ws in disconnected:
        if ws in session_state.websockets:
            session_state.websockets.remove(ws)

def sync_broadcast_event(event_data: Dict[str, Any]):
    """Synchronous wrapper for broadcasting from the agent thread."""
    # We need to run the async broadcast in the main event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_event(event_data), loop)
        else:
            # Fallback for when loop is not accessible (should not happen in FastAPI)
            logger.warning("Could not access event loop for broadcasting")
    except RuntimeError:
        # If no loop in this thread, try to find the main loop
        # This part is tricky. For now, we rely on the fact that FastAPI runs in an event loop.
        pass

# --- Agent Integration ---

def display_callback(event: Dict[str, Any]):
    """Callback for DisplayManager to pipe events to WebSocket."""
    # This runs in the agent thread
    # Use asyncio.run_coroutine_threadsafe with the stored server loop
    global server_loop
    if server_loop is None:
        logger.warning("Server loop not available, cannot broadcast event")
        return
    
    try:
        # Check if loop is still running
        if not server_loop.is_running():
            logger.warning("Server loop is not running")
            return
            
        future = asyncio.run_coroutine_threadsafe(broadcast_event(event), server_loop)
        # Don't wait for result to avoid blocking
    except Exception as e:
        logger.warning(f"Failed to broadcast event: {e}")

server_loop = None
_matplotlib_patch_state: Dict[str, Any] = {}


async def startup_event():
    global server_loop
    server_loop = asyncio.get_running_loop()
    
    # Configure DisplayManager to use our callback
    DisplayManager().set_event_callback(display_callback)
    logger.info("DisplayManager callback registered")
    
    # --- Monkeypatch matplotlib to capture plots ---
    if _matplotlib_patch_state.get("patched"):
        return
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    original_plt_savefig = plt.savefig
    original_fig_savefig = Figure.savefig
    
    # Track already-broadcast images to avoid duplicates
    _broadcast_images = set()
    
    def broadcast_image(fname):
        """Helper to broadcast an image event."""
        global output_dir_path
        if not fname:
            return

        try:
            import base64
            from pathlib import Path
            
            # Only handle common image formats
            if str(fname).lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                path = Path(fname).resolve()
                
                # Skip if already broadcast (deduplication)
                path_str = str(path)
                if path_str in _broadcast_images:
                    logger.debug(f"Skipping duplicate image: {path.name}")
                    return
                _broadcast_images.add(path_str)
                
                if path.exists():
                    logger.debug(f"Processing image: {path.name}, output_dir_path={output_dir_path}")
                    
                    # Always use base64 to ensure images display correctly
                    # (URL-based serving has path resolution issues on Windows)
                    src = None
                    if not str(fname).lower().endswith('.pdf'):
                        with open(path, "rb") as img_file:
                            b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        if str(fname).lower().endswith('.svg'):
                            mime = "image/svg+xml"
                        elif str(fname).lower().endswith('.png'):
                            mime = "image/png"
                        else:
                            mime = "image/jpeg"
                        src = f"data:{mime};base64,{b64_string}"
                    
                    if src:
                        # Broadcast event
                        event = {
                            "type": "image",
                            "data": {
                                "src": src,
                                "filename": path.name
                            }
                        }
                        if server_loop:
                            asyncio.run_coroutine_threadsafe(broadcast_event(event), server_loop)
                        logger.info(f"Captured image: {path.name}")
                else:
                    logger.warning(f"Image file does not exist: {path}")
        except Exception as e:
            logger.warning(f"Failed to capture image: {e}")

    
    def captured_plt_savefig(*args, **kwargs):
        # Call original
        original_plt_savefig(*args, **kwargs)
        fname = args[0] if args else kwargs.get('fname')
        broadcast_image(fname)

    def captured_fig_savefig(self, *args, **kwargs):
        # Call original
        original_fig_savefig(self, *args, **kwargs)
        fname = args[0] if args else kwargs.get('fname')
        broadcast_image(fname)

    plt.savefig = captured_plt_savefig
    Figure.savefig = captured_fig_savefig
    _matplotlib_patch_state.update(
        {
            "patched": True,
            "plt": plt,
            "figure_cls": Figure,
            "original_plt_savefig": original_plt_savefig,
            "original_fig_savefig": original_fig_savefig,
        }
    )
    logger.info("Monkeypatched matplotlib.pyplot.savefig and Figure.savefig for image capture")


async def shutdown_event():
    global server_loop
    controller = session_state.controller
    if controller is not None:
        try:
            controller.release_active_session()
        except Exception:
            logger.debug("Failed to release active session ownership on shutdown", exc_info=True)
    try:
        terminate_active_training_subprocesses()
    except Exception:
        logger.debug("Failed to terminate active training subprocesses on shutdown", exc_info=True)
    if _matplotlib_patch_state.get("patched"):
        try:
            _matplotlib_patch_state["plt"].savefig = _matplotlib_patch_state["original_plt_savefig"]
            _matplotlib_patch_state["figure_cls"].savefig = _matplotlib_patch_state["original_fig_savefig"]
        except Exception:
            logger.debug("Failed to restore matplotlib savefig hooks", exc_info=True)
        _matplotlib_patch_state.clear()
    DisplayManager().set_event_callback(None)
    server_loop = None


# --- Endpoints ---

from fastapi.staticfiles import StaticFiles
import os

# ... (Previous imports)

# --- Endpoints ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_state.websockets.append(websocket)
    try:
        while True:
            # Keep alive / listen for client events if needed
            data = await websocket.receive_text()
            # We can handle client-side pings or specific commands here
    except WebSocketDisconnect:
        if websocket in session_state.websockets:
            session_state.websockets.remove(websocket)

@app.post("/api/init")
async def initialize_session(request: InitRequest):
    """Initialize the agent session."""
    try:
        # Create user goal
        user_goal = {
            "raw_question": "",
            "requested_analyses": [],
            "device": request.device,
            "report_format": "html",
        }
        
        runtime_cfg = _resolve_llm_runtime_config(request)
        llm, _ = _instantiate_runtime_llm(runtime_cfg)
        cli_config = session_state.initial_config or {}
        
        # Initialize session
        # Use CLI config for paths if available, fallback to request
        input_path = request.input_path or cli_config.get("input_path")
        output_path = request.output_path or cli_config.get("output_path")
        if not output_path:
            output_path = f"{input_path}_output" if input_path else "cytobridge_output"
        output_path = _resolve_output_path(output_path)
        _mount_output_dir(output_path)
        
        session = InteractiveSession(
            input_path=input_path,
            user_goal=user_goal,
            llm=llm,
            output_dir=output_path,
            enable_multimodal=request.enable_multimodal
        )
        runtime_context_window = _sync_session_context_window(session, runtime_cfg)
        _apply_exposure_settings_to_state(session.state)
        activation_mode = cli_config.get("tool_activation_mode") or request.tool_activation_mode
        if activation_mode in {"auto_next_turn", "manual_review"}:
            session.state["tool_activation_mode"] = activation_mode
        harvest_enabled = cli_config.get("tool_harvest_enabled", request.tool_harvest_enabled)
        session.state["tool_harvest_enabled"] = bool(harvest_enabled)
        proposal_mode = _normalize_algorithm_proposal_review_mode(
            request.algorithm_proposal_review_mode
            or cli_config.get("algorithm_proposal_review_mode")
            or "agent_decide"
        )
        session.state["algorithm_proposal_review_mode"] = proposal_mode
        idea_mode = _normalize_idea_review_mode(
            request.idea_review_mode
            or cli_config.get("idea_review_mode")
            or "agent_decide"
        )
        session.state["idea_review_mode"] = idea_mode
        stop_hook_enabled = _normalize_stop_hook_enabled(
            request.stop_hook_enabled
            if request.stop_hook_enabled is not None
            else cli_config.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
        )
        stop_hook_mode = _normalize_stop_hook_mode(
            request.stop_hook_mode
            if request.stop_hook_mode is not None
            else cli_config.get("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
        )
        stop_hook_prompt = _normalize_stop_hook_prompt(
            request.stop_hook_prompt
            if request.stop_hook_prompt is not None
            else cli_config.get("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
        )
        stop_hook_max_triggers = _normalize_stop_hook_max_triggers(
            request.stop_hook_max_triggers
            if request.stop_hook_max_triggers is not None
            else cli_config.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
        )
        session.state["stop_hook_enabled"] = stop_hook_enabled
        session.state["stop_hook_mode"] = stop_hook_mode
        session.state["stop_hook_prompt"] = stop_hook_prompt
        session.state["stop_hook_max_triggers"] = stop_hook_max_triggers
        try:
            session.planner.refresh_tooling()
        except Exception:
            logger.debug("Failed to refresh tooling after init review-mode assignment", exc_info=True)

        with session_state.lock:
            session_state.active_session = session
            # Clear events log for new session
            session_state.events_log = []
            session_state.stop_requested = False
            session_state.agent_thread = None
            session_state.active_turn_id = None
            session_state.initial_config.update(
                {
                    "llm_model": runtime_cfg.get("model"),
                    "llm_provider": runtime_cfg.get("provider"),
                    "llm_base_url": runtime_cfg.get("base_url"),
                    "llm_auth_mode": runtime_cfg.get("auth_mode"),
                    "llm_profile_id": runtime_cfg.get("profile_id"),
                    "llm_thinking_level": runtime_cfg.get("thinking_level"),
                    "llm_context_window": runtime_context_window,
                    "allow_large_context_window": bool(runtime_cfg.get("allow_large_context_window", False)),
                    "input_path": input_path,
                    "output_path": output_path,
                    "algorithm_proposal_review_mode": proposal_mode,
                    "idea_review_mode": idea_mode,
                    "stop_hook_enabled": stop_hook_enabled,
                    "stop_hook_mode": stop_hook_mode,
                    "stop_hook_prompt": stop_hook_prompt,
                    "stop_hook_max_triggers": stop_hook_max_triggers,
                }
            )
            if request.openai_api_key:
                session_state.initial_config.update(
                    _provider_key_save_payload(str(runtime_cfg.get("provider") or "auto"), request.openai_api_key)
                )
            
        # Set stop check callback on planner
        def check_stop():
            return session_state.stop_requested
        session.planner.stop_check = check_stop
        _adopt_web_session(session, runtime_cfg)
            
        return {"status": "success", "message": "Session initialized"}
        
    except Exception as e:
        logger.exception("Failed to init session")
        return {"status": "error", "message": str(e)}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message to the agent."""
    slash_result = await _handle_slash_command(request.message, request.attachments or [])
    if slash_result is not None:
        return slash_result
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    if request.attachments and not session_state.active_session.enable_multimodal:
        return {"status": "error", "message": "Multimodal input is disabled for the current session"}
    return _start_agent_turn(
        user_message=request.message,
        attachments=request.attachments or [],
        log_user_message=True,
        client_request_id=request.client_request_id or "",
    )


def _start_agent_turn(
    user_message: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
    *,
    log_user_message: bool = True,
    client_request_id: str = "",
) -> Dict[str, Any]:
    """Run one agent turn in background and stream completion/error events."""
    active_session = session_state.active_session
    if not active_session:
        return {"status": "error", "message": "Session not initialized"}
    safe_attachments = list(attachments or [])
    turn_id = str(uuid.uuid4())
    normalized_request_id = _normalize_client_request_id(client_request_id)
    duplicate_record = _reserve_chat_request(normalized_request_id, turn_id)
    if duplicate_record:
        return _duplicate_chat_response(normalized_request_id, duplicate_record)
    if session_state.agent_thread and session_state.agent_thread.is_alive():
        _forget_chat_request(normalized_request_id)
        message = (
            "Agent is stopping. Please retry in a moment."
            if session_state.stop_requested
            else "Agent is busy. Please wait for current turn to finish."
        )
        return {"status": "error", "message": message}

    if log_user_message:
        attachment_events: List[Dict[str, Any]] = []
        for idx, item in enumerate(safe_attachments):
            mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip()
            file_name = str(item.get("file_name") or item.get("fileName") or f"image_{idx + 1}").strip() or f"image_{idx + 1}"
            content = str(item.get("content") or "").strip()
            if not mime_type or not content:
                continue
            attachment_events.append(
                {
                    "type": "image",
                    "mime_type": mime_type,
                    "file_name": file_name,
                    "data_url": f"data:{mime_type};base64,{content}",
                }
            )

        _record_session_event(
            runtime_event(
                "user_message",
                {"content": user_message, "attachments": attachment_events},
                turn_id=turn_id,
                process_id=turn_id,
                session_id=getattr(active_session, "session_id", None),
                source="web",
            )
        )

    def run_agent_turn() -> None:
        try:
            with DisplayManager().event_context(turn_id=turn_id, process_id=turn_id):
                controller = session_state.controller or _web_controller()
                if controller.active_session is not active_session:
                    _adopt_web_session(active_session, _resolve_llm_runtime_config())
                    controller = session_state.controller or controller
                result = controller.run_turn(user_message, attachments=safe_attachments)
                response = result.response
            turn_complete_event = runtime_event(
                "turn_complete",
                {"response": response},
                session_id=getattr(active_session, "session_id", None),
                turn_id=turn_id,
                process_id=turn_id,
                source="web",
            )
            if server_loop:
                future = asyncio.run_coroutine_threadsafe(
                    broadcast_event(turn_complete_event),
                    server_loop,
                )
                try:
                    # Ensure turn_complete is appended into events_log before snapshot save.
                    future.result(timeout=2)
                except Exception:
                    logger.debug("turn_complete broadcast wait failed", exc_info=True)
                    if (
                        not session_state.events_log
                        or session_state.events_log[-1].get("type") != "turn_complete"
                        or session_state.events_log[-1].get("turn_id") != turn_id
                    ):
                        _record_session_event(turn_complete_event)
            else:
                # Fallback: still record completion event for resume timeline.
                if (
                    not session_state.events_log
                    or session_state.events_log[-1].get("type") != "turn_complete"
                    or session_state.events_log[-1].get("turn_id") != turn_id
                ):
                    _record_session_event(turn_complete_event)

            # Persist one more snapshot so the final assistant turn_complete event survives restart/resume.
            try:
                active_session._save_conversation()
            except Exception:
                logger.debug("post-turn checkpoint save failed", exc_info=True)
            _mark_chat_request_status(normalized_request_id, "completed")
        except AgentStopRequested:
            logger.info("Agent turn interrupted by user stop request")
            try:
                abort = getattr(active_session, "abort_interrupted_turn", None)
                if callable(abort):
                    abort("Stopped by user.")
            except Exception:
                logger.debug("post-stop history abort cleanup failed", exc_info=True)
            try:
                active_session._save_conversation()
            except Exception:
                logger.debug("post-stop checkpoint save failed", exc_info=True)
            _mark_chat_request_status(normalized_request_id, "stopped")
        except KeyboardInterrupt:
            logger.info("Agent turn interrupted by KeyboardInterrupt")
            try:
                abort = getattr(active_session, "abort_interrupted_turn", None)
                if callable(abort):
                    abort("Interrupted by KeyboardInterrupt.")
            except Exception:
                logger.debug("post-interrupt history abort cleanup failed", exc_info=True)
            try:
                active_session._save_conversation()
            except Exception:
                logger.debug("post-interrupt checkpoint save failed", exc_info=True)
            _mark_chat_request_status(normalized_request_id, "interrupted")
        except Exception as e:
            logger.exception("Agent turn failed")
            _mark_chat_request_status(normalized_request_id, "failed")
            if server_loop:
                asyncio.run_coroutine_threadsafe(
                    broadcast_event(
                        {
                            "type": "error",
                            "turn_id": turn_id,
                            "process_id": turn_id,
                            "data": {"message": str(e)},
                        }
                    ),
                    server_loop,
                )
        finally:
            was_stop_requested = bool(session_state.stop_requested)
            with session_state.lock:
                if session_state.agent_thread is threading.current_thread():
                    session_state.agent_thread = None
                    session_state.active_turn_id = None
                    session_state.stop_requested = False
            if was_stop_requested:
                stopped_event = {
                    "type": "stopped",
                    "turn_id": turn_id,
                    "process_id": turn_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "message": "Agent execution stopped by user",
                        "interrupt_sent": True,
                        "thread_alive": False,
                    },
                }
                if server_loop:
                    asyncio.run_coroutine_threadsafe(broadcast_event(stopped_event), server_loop)
                else:
                    _record_session_event(stopped_event)

    session_state.stop_requested = False
    thread = threading.Thread(target=run_agent_turn)
    with session_state.lock:
        session_state.agent_thread = thread
        session_state.active_turn_id = turn_id
    thread.start()
    return {"status": "success", "message": "Agent is processing"}

@app.post("/api/stop")
async def stop_agent():
    """Stop the current agent execution."""
    if not session_state.active_session:
        return {"status": "error", "message": "No active session"}
    
    session_state.stop_requested = True
    thread = session_state.agent_thread
    turn_id = session_state.active_turn_id
    logger.info("Stop requested by user")
    terminated_training = await asyncio.to_thread(
        terminate_active_training_subprocesses,
        "api_stop",
    )
    if terminated_training:
        logger.info("Terminated active training subprocesses on stop: %s", terminated_training)

    interrupt_sent = False
    initial_thread_alive = bool(thread and thread.is_alive())
    thread_alive = initial_thread_alive
    if thread_alive:
        interrupt_sent = _raise_in_thread(thread, AgentStopRequested)
        await asyncio.to_thread(thread.join, 2.0)
        thread_alive = bool(thread.is_alive())
        if not thread_alive:
            with session_state.lock:
                if session_state.agent_thread is thread:
                    session_state.agent_thread = None
                    session_state.active_turn_id = None
                    session_state.stop_requested = False
    
    # If the worker finished during join, its finally block broadcasts the final
    # stopped event. The endpoint only emits immediately when there was no worker
    # or when the worker is still unwinding.
    if thread_alive or not initial_thread_alive:
        await broadcast_event({
            "type": "stopped",
            "turn_id": turn_id,
            "process_id": turn_id,
            "data": {
                "message": (
                    "Agent execution stopped by user"
                    if not thread_alive
                    else "Stop requested; agent is still unwinding"
                ),
                "interrupt_sent": interrupt_sent,
                "thread_alive": thread_alive,
                "terminated_training_subprocesses": terminated_training,
            },
        })
    
    return {
        "status": "success",
        "message": "Stop signal sent" if thread_alive else "Agent stopped",
        "thread_alive": thread_alive,
        "interrupt_sent": interrupt_sent,
        "terminated_training_subprocesses": terminated_training,
    }


@app.post("/api/new_session")
async def new_session():
    """Clear active in-memory session so frontend can start a fresh chat."""
    if session_state.agent_thread and session_state.agent_thread.is_alive():
        return {
            "status": "error",
            "message": "Agent is busy. Stop the current run before starting a new chat.",
        }

    with session_state.lock:
        session_state.active_session = None
        session_state.events_log = []
        session_state.stop_requested = False
        session_state.agent_thread = None
        session_state.active_turn_id = None

    return {"status": "success", "message": "Active session cleared"}


@app.post("/api/switch_model")
async def switch_model(request: ModelSwitchRequest):
    """Switch model for the current active session without losing conversation history."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}

    if session_state.agent_thread and session_state.agent_thread.is_alive():
        return {"status": "error", "message": "Agent is busy. Stop the current run before switching model."}

    try:
        current_cfg = _resolve_llm_runtime_config()
        new_cfg = _resolve_llm_runtime_config(request)
        llm, mode = _instantiate_runtime_llm(new_cfg)

        session = session_state.active_session
        proposal_mode = _normalize_algorithm_proposal_review_mode(request.algorithm_proposal_review_mode)
        if request.algorithm_proposal_review_mode is not None:
            session.state["algorithm_proposal_review_mode"] = proposal_mode
        idea_mode = _normalize_idea_review_mode(request.idea_review_mode)
        if request.idea_review_mode is not None:
            session.state["idea_review_mode"] = idea_mode
        if request.stop_hook_enabled is not None:
            session.state["stop_hook_enabled"] = _normalize_stop_hook_enabled(request.stop_hook_enabled)
        if request.stop_hook_mode is not None:
            session.state["stop_hook_mode"] = _normalize_stop_hook_mode(request.stop_hook_mode)
        if request.stop_hook_prompt is not None:
            session.state["stop_hook_prompt"] = _normalize_stop_hook_prompt(request.stop_hook_prompt)
        if request.stop_hook_max_triggers is not None:
            session.state["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(
                request.stop_hook_max_triggers,
                default=session.state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
            )
        restore_diag = session.update_llm(llm)
        runtime_context_window = _sync_session_context_window(session, new_cfg)
        try:
            session.planner.refresh_tooling()
        except Exception:
            logger.debug("Failed to refresh tooling after switch_model", exc_info=True)

        session_state.initial_config.update(
            {
                "llm_model": new_cfg.get("model"),
                "llm_provider": new_cfg.get("provider"),
                "llm_base_url": new_cfg.get("base_url"),
                "llm_auth_mode": new_cfg.get("auth_mode"),
                "llm_profile_id": new_cfg.get("profile_id"),
                "llm_thinking_level": new_cfg.get("thinking_level"),
                "llm_context_window": runtime_context_window,
                "allow_large_context_window": bool(new_cfg.get("allow_large_context_window", False)),
                "algorithm_proposal_review_mode": session.state.get(
                    "algorithm_proposal_review_mode",
                    proposal_mode,
                ),
                "idea_review_mode": session.state.get("idea_review_mode", idea_mode),
                "stop_hook_enabled": _normalize_stop_hook_enabled(
                    session.state.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
                ),
                "stop_hook_mode": _normalize_stop_hook_mode(
                    session.state.get("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
                ),
                "stop_hook_prompt": _normalize_stop_hook_prompt(
                    session.state.get("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
                ),
                "stop_hook_max_triggers": _normalize_stop_hook_max_triggers(
                    session.state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
                ),
            }
        )
        key_payload = _provider_key_save_payload(str(new_cfg.get("provider") or "auto"), request.openai_api_key)
        if key_payload:
            session_state.initial_config.update(key_payload)
        base_url_payload = _provider_base_url_save_payload(
            str(new_cfg.get("provider") or "auto"),
            new_cfg.get("base_url"),
        )
        if base_url_payload:
            session_state.initial_config.update(base_url_payload)

        if request.persist:
            to_save: Dict[str, Any] = {}
            if request.llm_model is not None and str(request.llm_model).strip():
                to_save["llm_model"] = request.llm_model.strip()
            if request.llm_provider is not None and str(request.llm_provider).strip():
                to_save["llm_provider"] = normalize_llm_provider(request.llm_provider)
            if request.llm_auth_mode is not None and str(request.llm_auth_mode).strip():
                to_save["llm_auth_mode"] = normalize_auth_mode(request.llm_auth_mode)
            if request.llm_profile_id is not None:
                to_save["llm_profile_id"] = (request.llm_profile_id or "").strip()
            should_persist_computed_xiaomi_off = (
                request.llm_thinking_level is None
                and request.llm_provider is not None
                and normalize_llm_provider(request.llm_provider) == "xiaomi"
                and new_cfg.get("thinking_level") == "off"
            )
            if request.llm_thinking_level is not None or should_persist_computed_xiaomi_off:
                to_save["llm_thinking_level"] = normalize_llm_reasoning_effort(
                    request.llm_thinking_level if request.llm_thinking_level is not None else new_cfg.get("thinking_level"),
                    default="low",
                )
            if request.llm_base_url is not None:
                to_save["llm_base_url"] = request.llm_base_url.strip()
                to_save.update(
                    _provider_base_url_save_payload(str(new_cfg.get("provider") or "auto"), request.llm_base_url)
                )
            if request.allow_large_context_window is not None:
                to_save["allow_large_context_window"] = bool(request.allow_large_context_window)
            if request.openai_api_key is not None and request.openai_api_key.strip():
                to_save.update(_provider_key_save_payload(str(new_cfg.get("provider") or "auto"), request.openai_api_key))
            if request.algorithm_proposal_review_mode is not None:
                to_save["algorithm_proposal_review_mode"] = proposal_mode
            if request.idea_review_mode is not None:
                to_save["idea_review_mode"] = idea_mode
            if request.stop_hook_enabled is not None:
                to_save["stop_hook_enabled"] = _normalize_stop_hook_enabled(request.stop_hook_enabled)
            if request.stop_hook_mode is not None:
                to_save["stop_hook_mode"] = _normalize_stop_hook_mode(request.stop_hook_mode)
            if request.stop_hook_prompt is not None:
                to_save["stop_hook_prompt"] = _normalize_stop_hook_prompt(request.stop_hook_prompt)
            if request.stop_hook_max_triggers is not None:
                to_save["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(
                    request.stop_hook_max_triggers,
                )
            if to_save:
                save_config(to_save)

        await broadcast_event(
            {
                "type": "model_switched",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "old_model": current_cfg.get("model"),
                    "new_model": new_cfg.get("model"),
                    "mode": mode,
                    "degraded": bool(restore_diag.get("degraded")),
                    "warnings": restore_diag.get("reasons", []),
                },
            }
        )
        return {
            "status": "success",
            "message": f"Model switched to {new_cfg.get('model')}.",
            "model": new_cfg.get("model"),
            "mode": mode,
            "restore_diagnostics": restore_diag,
        }
    except Exception as e:
        logger.exception("Failed to switch model")
        return {"status": "error", "message": str(e)}


@app.post("/api/stop_hook_settings")
async def update_stop_hook_settings(request: StopHookSettingsRequest):
    """Update stop-hook settings without rebuilding the LLM client."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    if session_state.agent_thread and session_state.agent_thread.is_alive():
        return {"status": "error", "message": "Agent is busy. Stop hook settings can be changed after the current run stops."}

    try:
        session = session_state.active_session
        updates: Dict[str, Any] = {}
        if request.stop_hook_enabled is not None:
            updates["stop_hook_enabled"] = _normalize_stop_hook_enabled(request.stop_hook_enabled)
        if request.stop_hook_mode is not None:
            updates["stop_hook_mode"] = _normalize_stop_hook_mode(request.stop_hook_mode)
        if request.stop_hook_prompt is not None:
            updates["stop_hook_prompt"] = _normalize_stop_hook_prompt(request.stop_hook_prompt)
        if request.stop_hook_max_triggers is not None:
            updates["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(
                request.stop_hook_max_triggers,
                default=session.state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
            )
        if updates:
            session.state.update(updates)
            session_state.initial_config.update(updates)
            save_config(updates)
            try:
                session.planner.refresh_tooling()
            except Exception:
                logger.debug("Failed to refresh tooling after stop-hook settings update", exc_info=True)
            try:
                session._save_conversation()
            except Exception:
                logger.debug("Failed to save conversation after stop-hook settings update", exc_info=True)
        settings = {
            "stop_hook_enabled": _normalize_stop_hook_enabled(
                session.state.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
            ),
            "stop_hook_mode": _normalize_stop_hook_mode(
                session.state.get("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
            ),
            "stop_hook_prompt": _normalize_stop_hook_prompt(
                session.state.get("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
            ),
            "stop_hook_max_triggers": _normalize_stop_hook_max_triggers(
                session.state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
            ),
        }
        await broadcast_event(
            {
                "type": "stop_hook_settings_updated",
                "timestamp": datetime.now().isoformat(),
                "data": settings,
            }
        )
        return {"status": "success", "message": "Stop hook settings updated.", **settings}
    except Exception as e:
        logger.exception("Failed to update stop hook settings")
        return {"status": "error", "message": str(e)}


@app.get("/api/config")
async def get_config():
    """Get the initial configuration and session status."""
    # Start with saved config from ~/.cellcompass
    config = get_saved_config()
    visibility = _resolve_exposure_settings()
    config.setdefault("tool_harvest_enabled", False)
    config.setdefault("algorithm_proposal_review_mode", "agent_decide")
    config.setdefault("idea_review_mode", "agent_decide")
    config.setdefault("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
    config.setdefault("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
    config.setdefault("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
    config.setdefault("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS)
    config.setdefault("llm_auth_mode", "auto")
    config.setdefault("llm_provider", "auto")
    config.setdefault("llm_profile_id", None)
    config.setdefault("llm_thinking_level", "low")
    config.setdefault("allow_large_context_window", False)
    config["hidden_skills"] = visibility["hidden_skills"]
    config["hidden_training_algorithms"] = visibility["hidden_training_algorithms"]
    # Override with initial_config if passed via CLI during this run
    if session_state.initial_config:
        config.update(session_state.initial_config)
    config["algorithm_proposal_review_mode"] = _normalize_algorithm_proposal_review_mode(
        config.get("algorithm_proposal_review_mode")
    )
    config["idea_review_mode"] = _normalize_idea_review_mode(config.get("idea_review_mode"))
    config["stop_hook_enabled"] = _normalize_stop_hook_enabled(
        config.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
    )
    config["stop_hook_mode"] = _normalize_stop_hook_mode(config.get("stop_hook_mode", DEFAULT_STOP_HOOK_MODE))
    config["stop_hook_prompt"] = _normalize_stop_hook_prompt(
        config.get("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
    )
    config["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(
        config.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
    )
    
    # Check if there's an active session
    if session_state.active_session:
        session = session_state.active_session
        _mount_output_dir(session.output_dir)
        config["hasActiveSession"] = True
        config["isAgentRunning"] = bool(
            session_state.agent_thread and session_state.agent_thread.is_alive()
        )
        config["sessionId"] = getattr(session, 'session_id', None)
        # Return conversation history for display
        config["conversationHistory"] = session.conversation_history
        # Return config values used by the session
        config["input_path"] = session.input_path
        config["output_path"] = session.output_dir
        # Return events log for timeline restoration
        config["eventsLog"] = session_state.events_log
        # Return latest resume diagnostics
        config["resumeDiagnostics"] = getattr(session, "resume_diagnostics", None)
        config["tool_activation_mode"] = session.state.get("tool_activation_mode", "auto_next_turn")
        config["tool_harvest_enabled"] = bool(session.state.get("tool_harvest_enabled", False))
        config["pending_tools"] = session.state.get("pending_tools", [])
        config["context_policy"] = dict(session.state.get("context_policy") or {})
        config["allow_large_context_window"] = bool(
            (config["context_policy"] or {}).get("allow_large_context_window", False)
        )
        config["llm_context_usage"] = dict(session.state.get("llm_context_usage") or {})
        config["algorithm_proposal_review_mode"] = _normalize_algorithm_proposal_review_mode(
            session.state.get("algorithm_proposal_review_mode")
        )
        config["idea_review_mode"] = _normalize_idea_review_mode(session.state.get("idea_review_mode"))
        config["stop_hook_enabled"] = _normalize_stop_hook_enabled(
            session.state.get("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
        )
        config["stop_hook_mode"] = _normalize_stop_hook_mode(
            session.state.get("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
        )
        config["stop_hook_prompt"] = _normalize_stop_hook_prompt(
            session.state.get("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
        )
        config["stop_hook_max_triggers"] = _normalize_stop_hook_max_triggers(
            session.state.get("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS),
        )
        config["hidden_skills"] = _normalize_hidden_skills(session.state.get("hidden_skills"))
        config["hidden_training_algorithms"] = _normalize_hidden_training_algorithms(
            session.state.get("hidden_training_algorithms")
        )
    else:
        config["hasActiveSession"] = False
        config["isAgentRunning"] = False

    if session_state.controller is not None:
        try:
            config["shared_runtime_status"] = session_state.controller.status()
        except Exception:
            config["shared_runtime_status"] = {"status": "error"}
        
    public_config = dict(config)
    provider_api_keys = _normalize_provider_api_keys(config.get("provider_api_keys"))
    current_provider = normalize_llm_provider(public_config.get("llm_provider") or "auto")
    public_config["provider_api_keys"] = {}
    public_config["has_provider_api_keys"] = {
        provider: True
        for provider in provider_api_keys.keys()
    }
    public_config["has_current_provider_api_key"] = bool(provider_api_keys.get(current_provider))
    if "openai_api_key" in public_config:
        public_config["openai_api_key"] = ""
    public_config["has_openai_api_key"] = bool(config.get("openai_api_key"))
    public_config["has_any_api_key"] = bool(config.get("openai_api_key") or provider_api_keys)
    try:
        from .utils.gemini_oauth import CREDENTIALS_FILE
        public_config["has_gemini_oauth_credentials"] = bool(CREDENTIALS_FILE.exists())
    except Exception:
        public_config["has_gemini_oauth_credentials"] = False
    codex_status = get_codex_status_safe()
    public_config["llm_provider_options"] = provider_options()
    public_config["has_codex_oauth_profiles"] = bool(codex_status.get("profiles"))
    public_config["codex_oauth_status"] = codex_status
    # Codex usage can be slow or temporarily unavailable. Do not block Web
    # initialization on it; the setup modal refreshes usage via a separate
    # endpoint after the page is interactive.
    public_config["codex_oauth_usage"] = {
        "provider": "openai-codex",
        "profiles": [],
        "status": "pending" if public_config["has_codex_oauth_profiles"] else "unavailable",
        "error": None if public_config["has_codex_oauth_profiles"] else "No Codex OAuth profile detected.",
    }
    return _json_response_safe(public_config)

@app.get("/api/auth/codex/usage")
async def get_codex_oauth_usage():
    """Return Codex OAuth usage without blocking initial Web configuration."""
    usage = await asyncio.to_thread(get_codex_usage_safe, all_profiles=True)
    usage = dict(usage) if isinstance(usage, dict) else {"provider": "openai-codex", "profiles": []}
    usage.setdefault("provider", "openai-codex")
    usage.setdefault("profiles", [])
    usage["status"] = "unavailable" if usage.get("error") else "loaded"
    return _json_response_safe({"status": "success", "usage": usage})

@app.get("/api/runtime/status")
async def get_runtime_status():
    """Return adapter-agnostic runtime status for Web, CLI, and supervisors."""
    controller = session_state.controller or _web_controller()
    return _json_response_safe(controller.status())


@app.get("/api/runtime/doctor")
async def get_runtime_doctor():
    """Return local runtime health checks."""
    from .runtime_diagnostics import run_doctor

    controller = session_state.controller or _web_controller()
    return _json_response_safe(run_doctor(store=controller.store, jobs=controller.jobs))


@app.get("/api/runtime/perf/{session_id}")
async def get_runtime_perf(session_id: str):
    """Return event-count and duration summary for a saved session."""
    from .runtime_diagnostics import session_perf_report

    controller = session_state.controller or _web_controller()
    return _json_response_safe(session_perf_report(session_id, store=controller.store))

@app.post("/api/config/save")
async def save_user_config(request: InitRequest):
    """Save user configuration to persistent storage ~/.cellcompass/config.json"""
    try:
        provider = normalize_llm_provider(request.llm_provider)
        new_config = {
            "input_path": request.input_path,
            "output_path": request.output_path,
            "llm_base_url": request.llm_base_url,
            "llm_provider": provider,
            "llm_auth_mode": normalize_auth_mode(request.llm_auth_mode),
            "llm_profile_id": request.llm_profile_id,
            "llm_thinking_level": normalize_llm_reasoning_effort(
                request.llm_thinking_level,
                default="low",
            ),
            "llm_model": request.llm_model,
            "allow_large_context_window": bool(request.allow_large_context_window),
            "device": request.device,
            "enable_multimodal": request.enable_multimodal,
            "tool_activation_mode": request.tool_activation_mode,
            "tool_harvest_enabled": bool(request.tool_harvest_enabled),
            "algorithm_proposal_review_mode": _normalize_algorithm_proposal_review_mode(
                request.algorithm_proposal_review_mode
            ),
            "idea_review_mode": _normalize_idea_review_mode(request.idea_review_mode),
            "stop_hook_enabled": _normalize_stop_hook_enabled(request.stop_hook_enabled),
            "stop_hook_mode": _normalize_stop_hook_mode(request.stop_hook_mode),
            "stop_hook_prompt": _normalize_stop_hook_prompt(request.stop_hook_prompt),
            "stop_hook_max_triggers": _normalize_stop_hook_max_triggers(
                request.stop_hook_max_triggers,
            ),
        }
        # Filter out None values, but allow explicit clearing for session-scoped stop-hook prompt text.
        new_config = {
            k: v
            for k, v in new_config.items()
            if v is not None and (v != "" or k in {"stop_hook_prompt"})
        }
        if request.openai_api_key is not None and request.openai_api_key.strip():
            new_config.update(_provider_key_save_payload(provider, request.openai_api_key))
        new_config.update(_provider_base_url_save_payload(provider, request.llm_base_url))
        save_config(new_config)
        
        # Update session_state default config in memory
        session_state.initial_config.update(new_config)
        return {"status": "success", "message": "Configuration saved persistently."}
    except Exception as e:
        logger.exception("Failed to save config")
        return {"status": "error", "message": str(e)}

# --- Conversation History API ---

@app.get("/api/conversations")
async def list_conversations():
    """List all saved conversations."""
    from .conversation_store import ConversationStore
    store = ConversationStore()
    conversations = store.list_conversations()
    return {"status": "success", "conversations": conversations}

@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Get a specific conversation."""
    from .conversation_store import ConversationStore
    store = ConversationStore()
    conversation = store.get_conversation(session_id, degrade_large_snapshot=True)
    if conversation:
        return {"status": "success", "conversation": conversation}
    return {"status": "error", "message": "Conversation not found"}

@app.delete("/api/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation."""
    from .conversation_store import ConversationStore
    store = ConversationStore()
    deleted = store.delete_conversation(session_id)
    if deleted:
        return {"status": "success", "message": "Conversation deleted"}
    return {"status": "error", "message": "Conversation not found"}

# Store output path for dynamic mount
output_dir_path = None
static_mounted = False


def _resolve_output_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return str(Path(path).expanduser().resolve())


def _mount_output_dir(path: Optional[str]) -> None:
    """Mount or remount /output to the requested directory."""
    global output_dir_path
    resolved = _resolve_output_path(path)
    if not resolved:
        return

    os.makedirs(resolved, exist_ok=True)
    if output_dir_path and Path(output_dir_path) == Path(resolved):
        return

    # Remove existing /output mount if present.
    app.router.routes = [
        r for r in app.router.routes
        if not (getattr(r, "path", None) == "/output" and getattr(r, "name", None) == "output_files")
    ]
    app.mount("/output", StaticFiles(directory=resolved), name="output_files")

    # Keep /output ahead of "/" static catch-all, otherwise /output/* returns 404.
    routes = list(app.router.routes)
    output_idx = next(
        (i for i, r in enumerate(routes) if getattr(r, "path", None) == "/output" and getattr(r, "name", None) == "output_files"),
        None,
    )
    static_idx = next(
        (
            i
            for i, r in enumerate(routes)
            if getattr(r, "name", None) == "static" and getattr(r, "path", None) in {"", "/"}
        ),
        None,
    )
    if output_idx is not None and static_idx is not None and output_idx > static_idx:
        output_route = routes.pop(output_idx)
        routes.insert(static_idx, output_route)
        app.router.routes = routes

    output_dir_path = resolved
    logger.info(f"Serving output files from {resolved} at /output")

def start_server(host="0.0.0.0", port=8000, initial_config=None):
    global output_dir_path, static_mounted
    if initial_config:
        # Filter out None values
        session_state.initial_config = {k: v for k, v in initial_config.items() if v is not None}
        
        # Store output path for image serving
        output_dir_path = _resolve_output_path(initial_config.get("output_path"))

        # Mount output directory FIRST (before static catch-all)
        if output_dir_path:
            _mount_output_dir(output_dir_path)
    
    # Mount static files LAST (catch-all for SPA)
    if not static_mounted:
        # Use path relative to this module (cytobridge_agent/web_server.py)
        # web/ is now inside CytoBridge-agent alongside cytobridge_agent/
        module_dir = Path(__file__).parent.parent  # CytoBridge-agent directory
        static_dir = module_dir / "web" / "dist"
        if static_dir.exists():
            app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
            logger.info(f"Serving static files from {static_dir}")
            static_mounted = True
        else:
            logger.warning(f"Static directory not found: {static_dir}. Run 'npm run build' in web/ directory.")

    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
    
@app.post("/api/resume/{session_id}")
async def resume_session(session_id: str):
    """Resume a previous session."""
    try:
        from .conversation_store import ConversationStore
        
        runtime_cfg = _resolve_llm_runtime_config()
        llm, _ = _instantiate_runtime_llm(runtime_cfg)
        
        # Load checkpoint data to restore events_log
        store = ConversationStore()
        checkpoint_data = store.get_conversation(session_id)
        if not checkpoint_data:
            return {"status": "error", "message": f"Session {session_id} not found"}
        
        session = InteractiveSession.from_checkpoint(session_id, llm)
        runtime_context_window = _sync_session_context_window(session, runtime_cfg)
        _mount_output_dir(session.output_dir)

        def check_stop():
            return session_state.stop_requested

        session.planner.stop_check = check_stop
        _adopt_web_session(session, runtime_cfg)
        
        with session_state.lock:
            session_state.active_session = session
            session_state.stop_requested = False
            session_state.agent_thread = None
            session_state.active_turn_id = None
            # Restore events log from checkpoint for timeline restoration
            if checkpoint_data and checkpoint_data.get("events_log"):
                session_state.events_log = checkpoint_data["events_log"]
            else:
                session_state.events_log = []

        diagnostics = session.resume_diagnostics or checkpoint_data.get("resume_diagnostics", {})
        if diagnostics.get("degraded"):
            await broadcast_event(
                {
                    "type": "resume_warning",
                    "timestamp": datetime.now().isoformat(),
                    "data": {"warnings": diagnostics.get("warnings", [])},
                }
            )

        return {
            "status": "success",
            "message": "Session resumed",
            "session_id": session_id,
            "resume_diagnostics": diagnostics,
            "llm_context_window": runtime_context_window,
        }
        
    except Exception as e:
        logger.exception("Failed to resume session")
        return {"status": "error", "message": str(e)}


@app.post("/api/compact/{session_id}")
async def compact_session_context(session_id: str):
    """Manually compact context for an active or persisted session."""
    try:
        from .conversation_store import ConversationStore

        if session_state.active_session and getattr(session_state.active_session, "session_id", None) == session_id:
            if session_state.agent_thread and session_state.agent_thread.is_alive():
                return {
                    "status": "error",
                    "message": "Cannot compact the active session while the agent is still running.",
                }
            compact_info = session_state.active_session.compact_context()
            if compact_info.get("changed"):
                await broadcast_event(
                    {
                        "type": "context_compacted",
                        "timestamp": datetime.now().isoformat(),
                        "data": {"source": "active_session", **compact_info},
                    }
                )
            return {"status": "success", "session_id": session_id, "compact_result": compact_info}

        store = ConversationStore()
        result = store.compact_conversation(session_id)
        if not result.get("ok"):
            return {"status": "error", "message": result.get("error", "Compaction failed")}
        return {"status": "success", "session_id": session_id, "compact_result": result}
    except Exception as e:
        logger.exception("Failed to compact session")
        return {"status": "error", "message": str(e)}


@app.get("/api/tools/catalog")
async def get_tool_catalog():
    """Get downstream tool catalog status for the active session."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        payload = tools_handler.get_tool_catalog_status()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {"raw": payload}
        return {"status": "success", "catalog": payload}
    except Exception as e:
        logger.exception("Failed to get tool catalog")
        return {"status": "error", "message": str(e)}


@app.get("/api/skills")
async def list_skills(scope: str = "planner"):
    """List available and currently loaded skills for planner or downstream scope."""
    try:
        selected = _normalize_skill_scope(scope)
        skills_tools, state = _resolve_session_skills_tools(selected)
        available_payload = json.loads(skills_tools.list_skills())
        loaded_payload = json.loads(skills_tools.show_loaded_skills())
        response = {
            "status": "success",
            "scope": selected,
            "available_skills": available_payload.get("skills", []),
            "loaded_skills": loaded_payload.get("skills", []),
            "skills_revision": int(loaded_payload.get("skills_revision", 0)),
            "hidden_skills": _normalize_hidden_skills(state.get("hidden_skills")),
            "hidden_training_algorithms": _normalize_hidden_training_algorithms(
                state.get("hidden_training_algorithms")
            ),
            "has_active_session": bool(session_state.active_session),
        }
        if selected == "planner":
            response["auto_skills"] = list(state.get("planner_active_auto_skills", []))
            response["workspace_policy"] = dict(state.get("planner_workspace_policy", {}))
            response["available_algorithms"] = _list_training_algorithms_with_visibility(state)
            response["research_ideas"] = _list_research_ideas_for_ui(state)
        return response
    except Exception as e:
        logger.exception("Failed to list skills")
        return {"status": "error", "message": str(e)}


@app.post("/api/skills/load")
async def load_skill_endpoint(request: SkillMutationRequest):
    """Load a skill into the active session for the given scope."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        selected = _normalize_skill_scope(request.scope)
        skills_tools, _ = _resolve_session_skills_tools(selected)
        message = skills_tools.load_skill(request.name)
        if message.startswith("❌"):
            return {"status": "error", "message": message}
        available_payload = json.loads(skills_tools.list_skills())
        loaded_payload = json.loads(skills_tools.show_loaded_skills())
        return {
            "status": "success",
            "scope": selected,
            "message": message,
            "available_skills": available_payload.get("skills", []),
            "loaded_skills": loaded_payload.get("skills", []),
        }
    except Exception as e:
        logger.exception("Failed to load skill")
        return {"status": "error", "message": str(e)}


@app.post("/api/skills/unload")
async def unload_skill_endpoint(request: SkillMutationRequest):
    """Unload a skill from the active session for the given scope."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        selected = _normalize_skill_scope(request.scope)
        skills_tools, _ = _resolve_session_skills_tools(selected)
        message = skills_tools.unload_skill(request.name)
        if message.startswith("❌"):
            return {"status": "error", "message": message}
        available_payload = json.loads(skills_tools.list_skills())
        loaded_payload = json.loads(skills_tools.show_loaded_skills())
        return {
            "status": "success",
            "scope": selected,
            "message": message,
            "available_skills": available_payload.get("skills", []),
            "loaded_skills": loaded_payload.get("skills", []),
        }
    except Exception as e:
        logger.exception("Failed to unload skill")
        return {"status": "error", "message": str(e)}


@app.post("/api/skills/exposure")
async def set_skill_exposure(request: SkillExposureRequest):
    """Hide/show a skill from agent prompt exposure, persisted across conversations."""
    try:
        selected = _normalize_skill_scope(request.scope)
        skills_tools, state = _resolve_session_skills_tools(selected)
        message = skills_tools.set_skill_exposed(request.name, bool(request.exposed))
        if message.startswith("❌"):
            return {"status": "error", "message": message}
        hidden_skills = _normalize_hidden_skills(state.get("hidden_skills"))
        if session_state.active_session:
            active_state = session_state.active_session.state
            active_state["hidden_skills"] = hidden_skills
            hidden_algorithms = _normalize_hidden_training_algorithms(
                active_state.get("hidden_training_algorithms")
            )
        else:
            hidden_algorithms = _resolve_exposure_settings()["hidden_training_algorithms"]
        _persist_exposure_settings(hidden_skills, hidden_algorithms)
        available_payload = json.loads(skills_tools.list_skills())
        loaded_payload = json.loads(skills_tools.show_loaded_skills())
        payload: Dict[str, Any] = {
            "status": "success",
            "scope": selected,
            "message": message,
            "available_skills": available_payload.get("skills", []),
            "loaded_skills": loaded_payload.get("skills", []),
            "hidden_skills": hidden_skills,
        }
        if selected == "planner":
            payload["available_algorithms"] = _list_training_algorithms_with_visibility(
                session_state.active_session.state if session_state.active_session else state
            )
            payload["research_ideas"] = _list_research_ideas_for_ui(
                session_state.active_session.state if session_state.active_session else state
            )
        return payload
    except Exception as e:
        logger.exception("Failed to set skill exposure")
        return {"status": "error", "message": str(e)}


@app.post("/api/algorithms/exposure")
async def set_algorithm_exposure(request: AlgorithmExposureRequest):
    """Hide/show custom training algorithms from agent prompt exposure."""
    try:
        name = str(request.name or "").strip().lower()
        if not name:
            return {"status": "error", "message": "Algorithm name is required."}

        if session_state.active_session:
            state = session_state.active_session.state
            hidden = set(_normalize_hidden_training_algorithms(state.get("hidden_training_algorithms")))
            hidden_skills = _normalize_hidden_skills(state.get("hidden_skills"))
        else:
            settings = _resolve_exposure_settings()
            hidden = set(_normalize_hidden_training_algorithms(settings.get("hidden_training_algorithms")))
            hidden_skills = _normalize_hidden_skills(settings.get("hidden_skills"))
            state = {
                "hidden_skills": hidden_skills,
                "hidden_training_algorithms": list(hidden),
            }

        if request.exposed:
            hidden.discard(name)
            msg = f"✅ Algorithm exposed: {name}"
        else:
            hidden.add(name)
            msg = f"✅ Algorithm hidden: {name}"

        hidden_algorithms = sorted(hidden)
        state["hidden_training_algorithms"] = hidden_algorithms
        if session_state.active_session:
            session_state.active_session.state["hidden_training_algorithms"] = hidden_algorithms
        _persist_exposure_settings(hidden_skills, hidden_algorithms)
        await broadcast_event(
            {
                "type": "algorithm_visibility_updated",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "algorithm": name,
                    "exposed": bool(request.exposed),
                    "scope": "planner",
                    "hidden_training_algorithms": hidden_algorithms,
                },
            }
        )
        return {
            "status": "success",
            "message": msg,
            "hidden_training_algorithms": hidden_algorithms,
            "available_algorithms": _list_training_algorithms_with_visibility(state),
        }
    except Exception as e:
        logger.exception("Failed to set algorithm exposure")
        return {"status": "error", "message": str(e)}


@app.get("/api/proposals/{algorithm_id}")
async def get_algorithm_proposal(algorithm_id: str):
    """Get persisted proposal payload for a specific algorithm id."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        raw = tools_handler.get_algorithm_proposal_status(algorithm_id=str(algorithm_id or ""))
        if isinstance(raw, str) and raw.startswith("Error:"):
            return {"status": "error", "message": raw}
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw}
        else:
            payload = raw
        return {"status": "success", "proposal": payload}
    except Exception as e:
        logger.exception("Failed to get algorithm proposal")
        return {"status": "error", "message": str(e)}


@app.get("/api/ideas")
async def list_research_ideas_endpoint():
    """List persisted research ideas for UI browsing."""
    try:
        state = session_state.active_session.state if session_state.active_session else {}
        return {
            "status": "success",
            "ideas": _list_research_ideas_for_ui(state),
            "active_research_idea_id": str(state.get("active_research_idea_id") or ""),
        }
    except Exception as e:
        logger.exception("Failed to list research ideas")
        return {"status": "error", "message": str(e)}


@app.get("/api/ideas/{idea_id}")
async def get_research_idea(idea_id: str):
    """Get persisted research idea payload for a specific idea id."""
    if not session_state.active_session:
        record = load_research_idea_record(str(idea_id or "").strip().lower())
        if record:
            return {"status": "success", "idea": record}
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        raw = tools_handler.get_research_idea_status(idea_id=str(idea_id or ""))
        if isinstance(raw, str) and raw.startswith("Error:"):
            return {"status": "error", "message": raw}
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"raw": raw}
        else:
            payload = raw
        return {"status": "success", "idea": payload}
    except Exception as e:
        logger.exception("Failed to get research idea")
        return {"status": "error", "message": str(e)}


@app.post("/api/ideas/review")
async def review_research_idea_endpoint(request: ResearchIdeaReviewRequest):
    """Apply user review decision to a research idea."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        active_session = session_state.active_session
        tools_handler = active_session.planner.tools_handler
        active_session.state["_idea_review_source"] = "user_api"
        try:
            msg = tools_handler.review_research_idea(
                idea_id=request.idea_id,
                decision=request.decision,
                reviewer_feedback=request.reviewer_feedback,
            )
        finally:
            active_session.state["_idea_review_source"] = ""
        if isinstance(msg, str) and msg.startswith("Error:"):
            return {"status": "error", "message": msg}
        latest_raw = tools_handler.get_research_idea_status(idea_id=request.idea_id)
        latest_payload: Dict[str, Any] = {}
        if isinstance(latest_raw, str):
            try:
                latest_payload = json.loads(latest_raw)
            except Exception:
                latest_payload = {"raw": latest_raw}
        elif isinstance(latest_raw, dict):
            latest_payload = latest_raw
        auto_continue_result: Dict[str, Any] = {"status": "skipped", "message": "auto_continue disabled"}
        if bool(request.auto_continue):
            followup = (
                "User has submitted research idea review via UI/API.\n"
                f"idea_id: {request.idea_id}\n"
                f"decision: {request.decision}\n"
                f"reviewer_feedback: {request.reviewer_feedback or '(none)'}\n"
                "Continue workflow accordingly."
            )
            auto_continue_result = _start_agent_turn(
                user_message=followup,
                attachments=[],
                log_user_message=False,
            )
        return {
            "status": "success",
            "message": msg,
            "idea": latest_payload,
            "auto_continue": auto_continue_result,
        }
    except Exception as e:
        logger.exception("Failed to review research idea")
        return {"status": "error", "message": str(e)}


@app.post("/api/ideas/active")
async def set_active_research_idea_endpoint(request: ActiveResearchIdeaRequest):
    """Set the active research idea for the current session."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        msg = tools_handler.set_active_research_idea(request.idea_id)
        if isinstance(msg, str) and msg.startswith("Error:"):
            return {"status": "error", "message": msg}
        return {
            "status": "success",
            "message": msg,
            "active_research_idea_id": str(session_state.active_session.state.get("active_research_idea_id") or ""),
            "ideas": _list_research_ideas_for_ui(session_state.active_session.state),
        }
    except Exception as e:
        logger.exception("Failed to set active research idea")
        return {"status": "error", "message": str(e)}


@app.post("/api/proposals/review")
async def review_algorithm_proposal(request: ProposalReviewRequest):
    """Apply user review decision to an algorithm proposal."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        active_session = session_state.active_session
        tools_handler = active_session.planner.tools_handler
        # Mark this review as an explicit user/API action (not model self-review).
        active_session.state["_proposal_review_source"] = "user_api"
        try:
            msg = tools_handler.review_algorithm_proposal(
                algorithm_id=request.algorithm_id,
                proposal_id=request.proposal_id,
                decision=request.decision,
                reviewer_feedback=request.reviewer_feedback,
            )
        finally:
            active_session.state["_proposal_review_source"] = ""
        if isinstance(msg, str) and msg.startswith("Error:"):
            return {"status": "error", "message": msg}
        latest_raw = tools_handler.get_algorithm_proposal_status(algorithm_id=request.algorithm_id)
        latest_payload: Dict[str, Any] = {}
        if isinstance(latest_raw, str):
            try:
                latest_payload = json.loads(latest_raw)
            except Exception:
                latest_payload = {"raw": latest_raw}
        elif isinstance(latest_raw, dict):
            latest_payload = latest_raw
        auto_continue_result: Dict[str, Any] = {"status": "skipped", "message": "auto_continue disabled"}
        if bool(request.auto_continue):
            followup = (
                "User has submitted proposal review via UI/API.\n"
                f"algorithm_id: {request.algorithm_id}\n"
                f"decision: {request.decision}\n"
                f"reviewer_feedback: {request.reviewer_feedback or '(none)'}\n"
                "Continue workflow accordingly."
            )
            auto_continue_result = _start_agent_turn(
                user_message=followup,
                attachments=[],
                log_user_message=False,
            )

        return {
            "status": "success",
            "message": msg,
            "proposal": latest_payload,
            "auto_continue": auto_continue_result,
        }
    except Exception as e:
        logger.exception("Failed to review algorithm proposal")
        return {"status": "error", "message": str(e)}


@app.post("/api/tools/activation_mode")
async def set_tool_activation_mode(request: ToolActivationModeRequest):
    """Set activation mode for generated tools."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        msg = tools_handler.set_tool_activation_mode(request.mode)
        await broadcast_event(
            {
                "type": "tool_activation_mode_updated",
                "timestamp": datetime.now().isoformat(),
                "data": {"mode": request.mode, "message": msg},
            }
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        logger.exception("Failed to set activation mode")
        return {"status": "error", "message": str(e)}


@app.post("/api/tools/harvest_mode")
async def set_tool_harvest_mode(request: ToolHarvestModeRequest):
    """Enable/disable downstream reusable-tool harvesting."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        msg = tools_handler.set_tool_harvest_enabled(enabled=request.enabled)
        await broadcast_event(
            {
                "type": "tool_harvest_mode_updated",
                "timestamp": datetime.now().isoformat(),
                "data": {"enabled": bool(request.enabled), "message": msg},
            }
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        logger.exception("Failed to set tool harvest mode")
        return {"status": "error", "message": str(e)}


@app.post("/api/tools/approve")
async def approve_pending_tools(request: ApproveToolsRequest):
    """Approve pending generated tools and activate eligible ones."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        msg = tools_handler.approve_pending_tools(tool_ids=request.tool_ids)
        await broadcast_event(
            {
                "type": "tool_review_updated",
                "timestamp": datetime.now().isoformat(),
                "data": {"action": "approve", "message": msg},
            }
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        logger.exception("Failed to approve tools")
        return {"status": "error", "message": str(e)}


@app.post("/api/tools/disable")
async def disable_generated_tool(request: DisableToolRequest):
    """Disable an active generated tool by name or tool_id."""
    if not session_state.active_session:
        return {"status": "error", "message": "Session not initialized"}
    try:
        tools_handler = session_state.active_session.planner.tools_handler
        msg = tools_handler.disable_generated_tool(request.name_or_id)
        await broadcast_event(
            {
                "type": "tool_review_updated",
                "timestamp": datetime.now().isoformat(),
                "data": {"action": "disable", "message": msg},
            }
        )
        return {"status": "success", "message": msg}
    except Exception as e:
        logger.exception("Failed to disable tool")
        return {"status": "error", "message": str(e)}
