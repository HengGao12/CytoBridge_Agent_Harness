from __future__ import annotations

import sys
from pathlib import Path


_CB_PATH = Path(__file__).resolve().parents[2] / "CytoBridge-main"
if _CB_PATH.exists() and str(_CB_PATH) not in sys.path:
    sys.path.insert(0, str(_CB_PATH))

from CytoBridge.tl.training_algorithm_registry import (  # noqa: F401
    DEFAULT_REQUIREMENTS_TEXT,
    REQUIRED_MANIFEST_FIELDS,
    LoadedTrainingAlgorithm,
    TrainingAlgorithmRecord,
    ValidationResult,
    default_training_algorithm_roots,
    list_training_algorithms,
    load_training_algorithm,
    render_training_algorithm_catalog_context,
    resolve_base_config,
    validate_training_algorithm,
)

__all__ = [
    "DEFAULT_REQUIREMENTS_TEXT",
    "REQUIRED_MANIFEST_FIELDS",
    "LoadedTrainingAlgorithm",
    "TrainingAlgorithmRecord",
    "ValidationResult",
    "default_training_algorithm_roots",
    "list_training_algorithms",
    "load_training_algorithm",
    "render_training_algorithm_catalog_context",
    "resolve_base_config",
    "validate_training_algorithm",
]
