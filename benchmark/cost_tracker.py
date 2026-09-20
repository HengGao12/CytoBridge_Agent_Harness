"""
Cost and efficiency tracker for benchmark agent runs.

Records wall-clock timing and API call metadata (tokens, cost estimates)
so each benchmark run can report Efficiency / Speed / Cost metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Cost estimation rules (USD per 1K tokens) ──
_COST_RATES: Dict[str, Dict[str, float]] = {
    "openai": {"input": 0.005, "output": 0.015},
    "codex": {"input": 0.005, "output": 0.015},
    "anthropic": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.005, "output": 0.015},
}


@dataclass
class ApiCallRecord:
    """A single API call with token counts and cost estimate."""
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_estimate: float


class CostTracker:
    """
    Tracks wall-clock elapsed time and API call costs for a single
    agent benchmark run.

    Usage::

        tracker = CostTracker()
        tracker.start()
        # ... agent runs, calls tracker.record_call(...) as needed ...
        tracker.stop()
        summary = tracker.to_dict()
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_seconds: float = 0.0
        self.api_calls: List[ApiCallRecord] = []
        self.total_cost_estimate: float = 0.0
        self._running: bool = False

    # ── Lifecycle ──

    def start(self) -> None:
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_seconds = 0.0
        self.api_calls.clear()
        self.total_cost_estimate = 0.0
        self._running = True

    def stop(self) -> None:
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed_seconds = self.end_time - self.start_time
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Recording ──

    def record_call(
        self,
        provider: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_estimate: Optional[float] = None,
    ) -> None:
        """Record a single API call.  If *cost_estimate* is not given, it is
        computed from *provider* and token counts using built-in rates."""
        if cost_estimate is None:
            cost_estimate = self._estimate_cost(provider, tokens_in, tokens_out)
        record = ApiCallRecord(
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_estimate=cost_estimate,
        )
        self.api_calls.append(record)
        self.total_cost_estimate += cost_estimate

    # ── Serialisation ──

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary dict."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "api_call_count": len(self.api_calls),
            "total_tokens_in": sum(c.tokens_in for c in self.api_calls),
            "total_tokens_out": sum(c.tokens_out for c in self.api_calls),
            "total_cost_estimate": round(self.total_cost_estimate, 6),
            "api_calls": [
                {
                    "provider": c.provider,
                    "model": c.model,
                    "tokens_in": c.tokens_in,
                    "tokens_out": c.tokens_out,
                    "cost_estimate": round(c.cost_estimate, 6),
                }
                for c in self.api_calls
            ],
        }

    # ── Helpers ──

    @staticmethod
    def _estimate_cost(provider: str, tokens_in: int, tokens_out: int) -> float:
        normalised = (provider or "").strip().lower()
        rates = _COST_RATES.get(normalised, _COST_RATES["default"])
        return (tokens_in / 1000.0) * rates["input"] + (tokens_out / 1000.0) * rates["output"]

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return (
            f"CostTracker({status}, elapsed={self.elapsed_seconds:.1f}s, "
            f"calls={len(self.api_calls)}, cost=${self.total_cost_estimate:.4f})"
        )
