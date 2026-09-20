"""Unified tool router for builtin and generated tools."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from .tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _normalize_identifier(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in name)
    if not sanitized:
        sanitized = "Tool"
    if sanitized[0].isdigit():
        sanitized = f"T_{sanitized}"
    return sanitized


class GeneratedExecutor:
    """Execute generated tools in a subprocess sandbox."""

    def __init__(self, output_dir: Path, timeout_sec: int = 45):
        self.output_dir = Path(output_dir)
        self.timeout_sec = timeout_sec

    def execute(self, spec: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
        code_path = spec.get("code_path")
        function_name = spec.get("function_name") or "run_tool"
        if not code_path:
            return f"❌ Generated tool has no code_path: {spec.get('name')}"

        path = Path(code_path)
        if not path.exists():
            return f"❌ Generated tool file missing: {path}"

        payload = json.dumps(kwargs or {}, ensure_ascii=False)
        runner = r'''
import importlib.util
import json
import pathlib
import sys
import traceback

mod_path = pathlib.Path(sys.argv[1]).resolve()
fn_name = sys.argv[2]
kwargs = json.loads(sys.argv[3])

try:
    spec = importlib.util.spec_from_file_location("generated_tool_module", str(mod_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    fn = getattr(module, fn_name)
    result = fn(**kwargs)
    try:
        json.dumps(result)
        safe_result = result
    except TypeError:
        safe_result = str(result)
    print(json.dumps({"ok": True, "result": safe_result}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{e.__class__.__name__}: {e}", "traceback": traceback.format_exc()}, ensure_ascii=False))
'''
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner, str(path), function_name, payload],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                cwd=str(self.output_dir),
            )
        except subprocess.TimeoutExpired:
            return f"❌ Generated tool timed out after {self.timeout_sec}s"
        except Exception as e:
            return f"❌ Generated tool execution failed: {e}"

        stdout = (proc.stdout or "").strip().splitlines()
        last_line = stdout[-1] if stdout else ""
        stderr = (proc.stderr or "").strip()

        try:
            result = json.loads(last_line) if last_line else {"ok": False, "error": "No output"}
        except Exception:
            tail = "\n".join(stdout[-20:]) if stdout else ""
            return f"❌ Generated tool returned non-JSON output.\nstdout:\n{tail}\nstderr:\n{stderr}"

        if not result.get("ok"):
            err = result.get("error", "Unknown error")
            tb = result.get("traceback", "")
            return f"❌ Generated tool failed: {err}\n{tb}".strip()

        output = result.get("result")
        if isinstance(output, (dict, list)):
            body = json.dumps(output, ensure_ascii=False, indent=2)
        else:
            body = str(output)

        if stderr:
            body = f"{body}\n\n[stderr]\n{stderr}"
        return body


class ToolRouter:
    """Build StructuredTool objects from a catalog."""

    def __init__(
        self,
        toolkit: Any,
        catalog: ToolCatalog,
        output_dir: Path,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.toolkit = toolkit
        self.catalog = catalog
        self.output_dir = Path(output_dir)
        self.event_sink = event_sink
        self.generated_executor = GeneratedExecutor(self.output_dir)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            logger.debug("ToolRouter event sink failed", exc_info=True)

    def _build_args_model(self, tool_name: str, schema: Dict[str, Any]) -> Optional[Type[BaseModel]]:
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict) or not properties:
            return None
        required = set(schema.get("required") or [])

        fields: Dict[str, Any] = {}
        for raw_name, prop in properties.items():
            name = raw_name if raw_name.isidentifier() else raw_name.replace("-", "_")
            if not name:
                continue

            prop = prop or {}
            typ = _TYPE_MAP.get(prop.get("type", "string"), Any)
            desc = prop.get("description", "")
            if raw_name in required and "default" not in prop:
                default = ...
            else:
                default = prop.get("default", None)
            fields[name] = (typ, Field(default=default, description=desc))

        if not fields:
            return None

        model_name = f"{_normalize_identifier(tool_name)}Input"
        return create_model(model_name, **fields)

    def _make_generated_callable(self, spec: Dict[str, Any]):
        def _call_generated(**kwargs):
            return self.generated_executor.execute(spec, kwargs)

        _call_generated.__name__ = f"call_{spec.get('name', 'generated_tool')}"
        _call_generated.__doc__ = spec.get("description", "Generated tool")
        return _call_generated

    def build_structured_tools(self) -> List[StructuredTool]:
        tools: List[StructuredTool] = []
        for spec in self.catalog.list_active_specs():
            kind = spec.get("kind")
            name = spec.get("name")
            if not name:
                continue

            if kind == "builtin":
                method_name = spec.get("method_name")
                if not method_name or not hasattr(self.toolkit, method_name):
                    logger.warning("Builtin method missing for tool %s: %s", name, method_name)
                    continue
                fn = getattr(self.toolkit, method_name)
                args_schema_hint = getattr(fn, "__tool_args_schema__", None)
                if args_schema_hint is None:
                    func_obj = getattr(fn, "__func__", None)
                    args_schema_hint = getattr(func_obj, "__tool_args_schema__", None) if func_obj is not None else None

                try:
                    tool = StructuredTool.from_function(
                        func=fn,
                        name=name,
                        description=spec.get("description") or fn.__doc__ or f"Execute {name}",
                        args_schema=args_schema_hint,
                    )
                except Exception as e:
                    logger.warning("Failed to build builtin tool %s: %s", name, e)
                    continue
                tools.append(tool)
                continue

            if kind == "generated":
                call_fn = self._make_generated_callable(spec)
                args_model = self._build_args_model(name, spec.get("args_schema") or {})
                try:
                    if args_model:
                        tool = StructuredTool.from_function(
                            func=call_fn,
                            name=name,
                            description=spec.get("description") or "Generated reusable tool",
                            args_schema=args_model,
                        )
                    else:
                        tool = StructuredTool.from_function(
                            func=call_fn,
                            name=name,
                            description=spec.get("description") or "Generated reusable tool",
                        )
                except Exception as e:
                    logger.warning("Failed to build generated tool %s: %s", name, e)
                    continue
                tools.append(tool)

        return tools
