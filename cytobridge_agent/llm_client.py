from __future__ import annotations

import os
from typing import List, Optional, Dict, Any

import httpx


class LLMClient:
    """
    Minimal OpenAI-compatible Chat Completions client.
    Works with OpenAI, vLLM/sglang, or any /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        base_url: Optional[str],
        model: Optional[str],
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout

    def is_enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Optional[str]:
        if not self.is_enabled():
            return None

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.base_url}/v1/chat/completions"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # OpenAI-compatible shape
                return data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - robust fallback
            print(f"[LLMClient] request failed: {exc}")
            return None

