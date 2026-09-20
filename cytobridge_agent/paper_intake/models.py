from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class DataAsset:
    """Downloadable or inspectable resource discovered from a manuscript."""

    url: str
    provider: str
    asset_kind: str
    file_format: str
    filename: str
    score: float = 0.0
    source: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ManuscriptInspection:
    """Structured manuscript parsing output written for the workflow agent."""

    input_source: str
    input_type: str
    title: str = ""
    doi: str = ""
    page_count: int = 0
    completeness_score: float = 0.0
    section_flags: Dict[str, bool] = field(default_factory=dict)
    discovered_urls: List[str] = field(default_factory=list)
    candidate_article_urls: List[str] = field(default_factory=list)
    accessions: Dict[str, List[str]] = field(default_factory=dict)
    relevant_sentences: List[str] = field(default_factory=list)
    primary_text_path: str = ""
    article_text_path: str = ""
    summary_path: str = ""
    manifest_path: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
