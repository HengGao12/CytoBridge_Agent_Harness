from __future__ import annotations

from typing import Optional


def effective_timeout_for_agent(agent: str | None, timeout_sec: Optional[float]) -> Optional[float]:
    """Return the externally enforced wall-clock timeout for a benchmark subprocess.

    User-requested policy: CytoBridge benchmark runs should not be hard-killed by the
    outer batch wrappers. The agent may take a long time but should be allowed to finish.
    Other agent runners keep their explicit timeout behavior unchanged.
    """
    if timeout_sec is None:
        return None
    normalized_agent = str(agent or "").strip().lower()
    if normalized_agent == "cytobridge":
        return None
    return float(timeout_sec)
