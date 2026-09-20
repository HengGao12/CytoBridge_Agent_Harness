"""
Abstract base class for benchmark agent runners.

Each agent (CytoBridge, Codex, Biomni, etc.) implements this interface
so the benchmark harness can drive them uniformly.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.cost_tracker import CostTracker


@dataclass
class RunResult:
    """Result of a single benchmark run."""

    success: bool
    predicted_h5ad: Optional[str] = None  # path to predicted_heldout.h5ad
    runtime_sec: float = 0.0
    error_message: Optional[str] = None
    agent_state: Optional[Dict[str, Any]] = None  # full state if available
    logs: List[str] = field(default_factory=list)
    native_status: str = "unknown"  # native_complete | native_incomplete | native_error
    format_status: str = "unknown"  # native_complete | recovered_complete | incomplete | error
    recovery_method: str = "none"  # none | workspace_copy | checkpoint_recovery | intermediate_conversion | ...
    recovery_status: str = "not_applied"  # not_applied | succeeded | failed
    artifact_origin: str = "native"  # native | recovered | mixed | none
    recovery_manifest: Optional[str] = None
    native_error_message: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    cost_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "success": self.success,
            "predicted_h5ad": self.predicted_h5ad,
            "runtime_sec": self.runtime_sec,
            "error_message": self.error_message,
            "logs": self.logs,
            "native_status": self.native_status,
            "format_status": self.format_status,
            "recovery_method": self.recovery_method,
            "recovery_status": self.recovery_status,
            "artifact_origin": self.artifact_origin,
            "recovery_manifest": self.recovery_manifest,
            "native_error_message": self.native_error_message,
            "extra_metadata": self.extra_metadata,
        }
        if self.cost_summary is not None:
            d["cost_summary"] = self.cost_summary
        return d


class AgentRunner(ABC):
    """
    Abstract interface for running an agent on a benchmark task.

    Subclasses must implement `run()` which takes:
    - a training h5ad path (held-out time point removed)
    - a standardized task prompt
    - an output directory
    - a random seed
    - device specification

    And returns a RunResult.
    """

    agent_id: str = "base"

    @abstractmethod
    def run(
        self,
        train_h5ad: str | Path,
        task_prompt: str,
        output_dir: str | Path,
        seed: int = 42,
        device: str = "cpu",
        time_key: str = "Time Point",
        timeout_sec: Optional[float] = None,
        cost_tracker: Optional[CostTracker] = None,
        **kwargs,
    ) -> RunResult:
        """
        Execute the agent on a benchmark task.

        Parameters
        ----------
        train_h5ad : path to training data (held-out removed)
        task_prompt : standardized task description
        output_dir : where agent should write outputs
        seed : random seed for reproducibility
        device : compute device (cpu / cuda / mps)
        time_key : column name for time labels in the adata
        timeout_sec : max runtime in seconds (None = no limit)
        cost_tracker : optional CostTracker to record API usage during the run

        Returns
        -------
        RunResult with success status, prediction path, timing, etc.
        """
        ...
    @staticmethod
    def _find_predicted_h5ad(output_dir: str | Path) -> Optional[str]:
        """
        Search output_dir for predicted_heldout.h5ad.
        Returns the path if found, None otherwise.
        """
        output_dir = Path(output_dir)
        # Primary expected name
        primary = output_dir / "predicted_heldout.h5ad"
        if primary.exists():
            return str(primary)
        # Search recursively
        for p in output_dir.rglob("predicted_heldout.h5ad"):
            return str(p)
        return None

    @staticmethod
    def _run_deterministic_fallback(
        *,
        output_dir: str | Path,
        workspace_dir: str | Path,
        logs: list[str],
    ) -> "bool":
        """DISABLED: Deterministic fallback removed to expose true agent capability.

        Returns False unconditionally -- agents must produce their own outputs.
        """
        logs.append(
            "DISABLED: deterministic fallback is not available. "
            "Agent must produce all 6 output files from its own pipeline."
        )
        return False
