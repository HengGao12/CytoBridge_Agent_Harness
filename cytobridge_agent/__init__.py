"""CellCompass / CytoBridge agent runtime package.

The public runtime entrypoint is the CLI (`cellcompass`) or the shared
`SessionController` API. Historical conversation snapshots remain readable, but
the old graph-level `run_agent` function is no longer exported as the package
API.

Keep this module lightweight. Subcommands such as `cellcompass auth` must be
able to import the package without pulling in scanpy/scipy/torch runtime stacks.
"""

import os
import tempfile
from pathlib import Path


def _default_cache_dir(name: str) -> str:
    for root in (Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"), Path(tempfile.gettempdir())):
        path = root / "cellcompass" / name
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return str(path)
        except Exception:
            continue
    return tempfile.gettempdir()


os.environ.setdefault("NUMBA_CACHE_DIR", _default_cache_dir("numba"))
os.environ.setdefault("MPLCONFIGDIR", _default_cache_dir("matplotlib"))

__version__ = "1.0.0"

_LAZY_EXPORTS = {
    "SessionController": ("session_controller", "SessionController"),
    "TurnResult": ("session_controller", "TurnResult"),
    "UserGoal": ("schemas", "UserGoal"),
    "DataSummary": ("schemas", "DataSummary"),
    "AnalysisScope": ("schemas", "AnalysisScope"),
    "PlanDecision": ("schemas", "PlanDecision"),
    "PilotResult": ("schemas", "PilotResult"),
    "DownstreamResult": ("schemas", "DownstreamResult"),
    "InsightReport": ("schemas", "InsightReport"),
    "FinalArtifacts": ("schemas", "FinalArtifacts"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value

__all__ = [
    # Runtime entry points
    "SessionController",
    "TurnResult",
    # Schemas
    "UserGoal",
    "DataSummary",
    "AnalysisScope",
    "PlanDecision",
    "PilotResult",
    "DownstreamResult",
    "InsightReport",
    "FinalArtifacts",
]
