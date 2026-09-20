"""Paper-to-AnnData intake helpers for CytoBridge workflows."""

from __future__ import annotations

from typing import Any

__all__ = [
    "inspect_paper_source",
    "prepare_anndata_from_paper",
]


def __getattr__(name: str) -> Any:
    if name in {"inspect_paper_source", "prepare_anndata_from_paper"}:
        from .pipeline import inspect_paper_source, prepare_anndata_from_paper

        exports = {
            "inspect_paper_source": inspect_paper_source,
            "prepare_anndata_from_paper": prepare_anndata_from_paper,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
