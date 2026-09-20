import json
import logging
import socket
import time
from typing import Any, Dict, List, Optional, Callable, Tuple
import urllib.request
import urllib.error
import uuid
import re

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall
)
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger("cytobridge_agent.cloud_code_chat")
CLIENT_METADATA = "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI"
THOUGHT_SIGNATURE_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _uppercase_schema_types(schema: Any) -> Any:
    """Recursively uppercase JSON schema ``type`` values for Cloud Code Assist."""
    if isinstance(schema, list):
        return [_uppercase_schema_types(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            if isinstance(value, str):
                out[key] = value.upper()
            elif isinstance(value, list):
                out[key] = [v.upper() if isinstance(v, str) else v for v in value]
            else:
                out[key] = value
        else:
            out[key] = _uppercase_schema_types(value)
    return out


def _normalize_system_instruction(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return {"role": "user", "parts": [{"text": text}]}
    if not isinstance(value, dict):
        return None

    sys = dict(value)
    sys["role"] = "user"
    parts = sys.get("parts")
    if isinstance(parts, list):
        normalized_parts = []
        for part in parts:
            if isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    normalized_parts.append({"text": part["text"]})
                elif "inlineData" in part:
                    normalized_parts.append({"inlineData": part["inlineData"]})
        if normalized_parts:
            sys["parts"] = normalized_parts
            return sys
    return None


def _sanitize_tool_schema(schema: Any, *, inside_properties: bool = False) -> Any:
    """Strips unsupported JSON Schema keywords for Google Cloud Code Assist."""
    unsupported = {
        "patternProperties", "additionalProperties", "$schema", "$id", "$ref", "$defs",
        "definitions", "examples", "minLength", "maxLength", "minimum", "maximum",
        "multipleOf", "pattern", "format", "minItems", "maxItems", "uniqueItems",
        "minProperties", "maxProperties"
    }

    if isinstance(schema, list):
        return [_sanitize_tool_schema(item, inside_properties=inside_properties) for item in schema]
    elif isinstance(schema, dict):
        sanitized = {}
        for k, v in schema.items():
            if not inside_properties and k in unsupported:
                continue
            child_inside_properties = k == "properties"
            sanitized[k] = _sanitize_tool_schema(v, inside_properties=child_inside_properties)

        # If some property definitions were removed during sanitization, keep
        # `required` aligned with the remaining property names.
        props = sanitized.get("properties")
        req = sanitized.get("required")
        if isinstance(props, dict) and isinstance(req, list):
            sanitized["required"] = [name for name in req if isinstance(name, str) and name in props]
        return sanitized
    return schema


def _extract_thought_signature(record: Dict[str, Any]) -> Optional[str]:
    candidate = (
        record.get("thinkingSignature")
        or record.get("signature")
        or record.get("thought_signature")
        or record.get("thoughtSignature")
    )
    if not isinstance(candidate, str):
        return None
    trimmed = candidate.strip()
    if not trimmed:
        return None
    if trimmed.startswith("msg_"):
        return None
    if len(trimmed) % 4 != 0:
        return None
    if not THOUGHT_SIGNATURE_RE.match(trimmed):
        return None
    return trimmed


def _tool_call_history_text(name: str, args: Any) -> str:
    try:
        rendered_args = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered_args = str(args)
    return f"[Historical tool call] {name} args={rendered_args}"


def _tool_response_history_text(name: str, result: Any) -> str:
    if isinstance(result, str):
        rendered = result
    else:
        try:
            rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
        except Exception:
            rendered = str(result)
    return f"[Historical tool result] {name} result={rendered}"


class ChatCloudCodeAssist(BaseChatModel):
    """Custom LangChain chat model that interacts directly with Google Cloud Code Assist."""
    
    model_name: str = "gemini-3-pro-preview"
    api_key: str = ""  # This is the OAuth access_token
    base_url: str = "https://cloudcode-pa.googleapis.com"
    temperature: float = 0.0
    project_id: str = "none"
    token_provider: Optional[Callable[[], str]] = None
    request_timeout_sec: float = 120.0
    max_retries: int = 5
    retry_backoff_sec: float = 1.5
    retry_all_http_errors: bool = True
    session_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "cloud-code-assist"

    def _format_messages(self, messages: List[BaseMessage]) -> Tuple[List[Dict], Optional[Dict[str, Any]]]:
        """Convert LangChain messages into Cloud Code Assist contents + systemInstruction."""
        system_instructions = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instructions.append(msg.content)

        formatted_contents = []
        signed_tool_call_ids = set()
        signed_tool_call_names = set()

        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue

            role = "user"
            if isinstance(msg, AIMessage):
                role = "model"
            elif isinstance(msg, ToolMessage):
                role = "user"

            parts = []
            raw_parts = None
            if isinstance(msg, AIMessage):
                raw_parts = (getattr(msg, "additional_kwargs", {}) or {}).get("cloud_code_content_parts")

            if isinstance(raw_parts, list):
                for raw_part in raw_parts:
                    if not isinstance(raw_part, dict):
                        continue
                    if "text" in raw_part and isinstance(raw_part.get("text"), str):
                        parts.append({"text": raw_part["text"]})
                        continue
                    if "inlineData" in raw_part:
                        parts.append({"inlineData": raw_part["inlineData"]})
                        continue
                    raw_call = raw_part.get("functionCall")
                    if isinstance(raw_call, dict):
                        name = str(raw_call.get("name", "")).strip()
                        args = raw_call.get("args", {})
                        signature = _extract_thought_signature(raw_call)
                        if name and signature:
                            outbound_call = {
                                "name": name,
                                "args": args,
                                "thoughtSignature": signature,
                            }
                            parts.append({"functionCall": outbound_call})
                            raw_id = raw_call.get("id")
                            if isinstance(raw_id, str) and raw_id.strip():
                                signed_tool_call_ids.add(raw_id)
                            signed_tool_call_names.add(name)
                        elif name:
                            parts.append({"text": _tool_call_history_text(name, args)})
                # Do not separately re-append msg.content or msg.tool_calls when raw provider parts exist.
            else:
                if not isinstance(msg, ToolMessage):
                    if isinstance(msg.content, str) and msg.content:
                        parts.append({"text": msg.content})
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    parts.append({"text": block.get("text", "")})
                                elif block.get("type") == "image_url":
                                    image_url = block.get("image_url", {}).get("url", "")
                                    if image_url.startswith("data:image/"):
                                        header, b64_data = image_url.split(",", 1)
                                        mime_type = header.split(":")[1].split(";")[0]
                                        parts.append({
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": b64_data
                                            }
                                        })

                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        name = str(tool_call.get("name", "")).strip()
                        if not name:
                            continue
                        parts.append({"text": _tool_call_history_text(name, tool_call.get("args", {}))})

            if isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                has_signed_context = (
                    (isinstance(tool_call_id, str) and tool_call_id in signed_tool_call_ids)
                    or msg.name in signed_tool_call_names
                )
                if has_signed_context:
                    tool_response: Dict[str, Any] = {
                        "name": msg.name,
                        "response": {
                            "result": msg.content
                        }
                    }
                    parts.append({
                        "functionResponse": tool_response
                    })
                else:
                    parts.append({"text": _tool_response_history_text(msg.name, msg.content)})

            if parts:
                formatted_contents.append({"role": role, "parts": parts})

        system_instruction = None
        if system_instructions:
            system_instruction = {
                "role": "user",
                "parts": [{"text": "\n".join(system_instructions)}],
            }

        return formatted_contents, system_instruction

    def _format_tools(self, kwargs: Dict[str, Any]) -> List[Dict]:
        tools = kwargs.get("tools", [])
        if not tools:
            return []

        formatted_tools = []
        for t in tools:
            if hasattr(t, "name") and hasattr(t, "description"):
                # Langchain Tool object
                schema = t.args_schema.schema() if t.args_schema else {}
            elif isinstance(t, dict) and "function" in t:
                # OpenAI formatted tool
                schema = t["function"].get("parameters", {})
                t = type('obj', (object,), {'name': t["function"]["name"], 'description': t["function"].get("description", "")})
            else:
                continue

            # Strip invalid keywords
            sanitized_schema = _sanitize_tool_schema(schema)
            sanitized_schema = _uppercase_schema_types(sanitized_schema)

            formatted_tools.append({
                "functionDeclarations": [{
                    "name": t.name,
                    "description": t.description,
                    "parameters": sanitized_schema
                }]
            })
            
        return formatted_tools

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        contents, system_instruction = self._format_messages(messages)
        tools = self._format_tools(kwargs)

        request_payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            }
        }
        if system_instruction is not None:
            request_payload["systemInstruction"] = system_instruction
        if "system_instruction" in kwargs and "systemInstruction" not in request_payload:
            normalized = _normalize_system_instruction(kwargs.get("system_instruction"))
            if normalized is not None:
                request_payload["systemInstruction"] = normalized
        if tools:
            request_payload["tools"] = tools
            request_payload["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": "VALIDATED",
                }
            }

        request_payload["systemInstruction"] = _normalize_system_instruction(
            request_payload.get("systemInstruction")
        )
        if request_payload.get("systemInstruction") is None:
            request_payload.pop("systemInstruction", None)

        if not self.session_id:
            self.session_id = f"cellcompass-{uuid.uuid4().hex}"
        request_payload["sessionId"] = self.session_id

        payload = {
            "project": self.project_id,
            "model": self.model_name,
            "request": request_payload,
            "requestType": "agent",
            "userAgent": "google-gemini-cli",
            "requestId": "agent-" + uuid.uuid4().hex,
        }

        url = f"{self.base_url}/v1internal:generateContent"
        token = self._get_bearer_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "google-api-nodejs-client/9.15.1",
            "X-Goog-Api-Client": "gl-node/22.17.0",
            "Client-Metadata": CLIENT_METADATA,
        }

        resp_json = self._post_with_retry(url=url, payload=payload, headers=headers)
        resp_root = resp_json.get("response", resp_json)

        candidates = resp_root.get("candidates", [])
        if not candidates:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        first_candidate = candidates[0]
        content_parts = first_candidate.get("content", {}).get("parts", [])
        
        text_content = ""
        tool_calls = []

        for idx, part in enumerate(content_parts):
            if "text" in part:
                text_content += part["text"]
            elif "functionCall" in part:
                tc = part["functionCall"]
                tc_id = tc.get("id") or f"call_{tc['name']}_{uuid.uuid4().hex[:10]}_{idx}"
                tool_calls.append(
                    ToolCall(
                        name=tc["name"],
                        args=tc.get("args", {}),
                        id=tc_id
                    )
                )

        ai_message = AIMessage(
            content=text_content,
            tool_calls=tool_calls,
            additional_kwargs={
                "cloud_code_content_parts": content_parts,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "ChatCloudCodeAssist":
        """Bind tools to the model."""
        return super().bind(tools=tools, **kwargs)

    def _get_bearer_token(self) -> str:
        if self.token_provider:
            return self.token_provider()
        return self.api_key

    def _post_with_retry(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        def _request(token: str) -> Dict[str, Any]:
            req_headers = dict(headers)
            req_headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers=req_headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=float(self.request_timeout_sec)) as response:
                return json.loads(response.read().decode('utf-8'))

        token = self._get_bearer_token()
        max_attempts = max(1, int(self.max_retries))
        transient_http_codes = {408, 429, 500, 502, 503, 504}

        for attempt in range(1, max_attempts + 1):
            try:
                return _request(token)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8')

                # OAuth token expiry: refresh then retry.
                if e.code == 401 and self.token_provider:
                    logger.info("Cloud Code token may be expired; refreshing token.")
                    token = self.token_provider()
                    if attempt < max_attempts:
                        continue
                    raise Exception(f"ChatCloudCodeAssist Failed after token refresh: {err_body}")

                should_retry_http = self.retry_all_http_errors or e.code in transient_http_codes
                if should_retry_http and attempt < max_attempts:
                    sleep_s = float(self.retry_backoff_sec) * (2 ** (attempt - 1))
                    logger.warning(
                        "Cloud Code HTTP %s (attempt %s/%s). Retrying in %.1fs.",
                        e.code,
                        attempt,
                        max_attempts,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue

                logger.error("Cloud Code API Error %s: %s", e.code, err_body)
                raise Exception(f"ChatCloudCodeAssist Failed: {err_body}")
            except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
                if attempt < max_attempts:
                    sleep_s = float(self.retry_backoff_sec) * (2 ** (attempt - 1))
                    logger.warning(
                        "Cloud Code network error (attempt %s/%s): %s. Retrying in %.1fs.",
                        attempt,
                        max_attempts,
                        e,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise Exception(f"ChatCloudCodeAssist network failure: {e}")

        raise Exception("ChatCloudCodeAssist request failed after retries.")
