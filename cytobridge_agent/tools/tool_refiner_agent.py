"""LLM-based refiner for generated tool candidates."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.llm_runtime import invoke_with_retry

logger = logging.getLogger(__name__)


class ToolRefinerAgent:
    """Refine candidate tool code to maximize reusability."""

    def __init__(self, llm: Any):
        self.llm = llm

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}

        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        for block in fenced:
            try:
                return json.loads(block)
            except Exception:
                continue

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}

    def refine(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm:
            return candidate

        payload = json.dumps(candidate, ensure_ascii=False, indent=2)
        prompt = f"""
Refine the following candidate reusable tool.

STRICT REQUIREMENTS:
1. Keep tool data-agnostic and reusable.
2. Remove dataset-specific constants, hardcoded paths, and project-specific column names.
3. Keep behavior deterministic and safe.
4. Return valid Python function code.
5. Include a minimal JSON schema for tool args.

Return JSON only with keys:
- name
- description
- function_name
- args_schema
- code
- usage
- rationale
- minimal_tests

Candidate:
{payload}
        """.strip()

        try:
            resp = invoke_with_retry(
                self.llm,
                [
                    SystemMessage(content="You are a strict software engineer refining reusable tools."),
                    HumanMessage(content=prompt),
                ],
                logger,
                "tool_refiner",
            )
            parsed = self._extract_json(getattr(resp, "content", ""))
            if not parsed:
                return candidate

            refined = dict(candidate)
            refined.update(parsed)
            return refined
        except Exception:
            logger.debug("Tool refinement failed, fallback to original candidate", exc_info=True)
            return candidate
