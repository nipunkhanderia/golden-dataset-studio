"""Core Pydantic models for Golden Dataset."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _short_id() -> str:
    return str(uuid.uuid4())[:8]


class GoldenEntry(BaseModel):
    """A single question-answer pair in the golden dataset."""

    id: str = Field(default_factory=_short_id)
    question: str = Field(...)
    answer: str = Field(...)
    contexts: List[str] = Field(default_factory=list)
    ground_truth: Optional[str] = Field(None)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def effective_ground_truth(self) -> str:
        return self.ground_truth if self.ground_truth is not None else self.answer


class DatasetVersion(BaseModel):
    """Immutable snapshot metadata for a committed dataset version."""

    version: str
    created_at: datetime = Field(default_factory=_utcnow)
    description: str = ""
    entry_count: int
    sha256: str
    parent_version: Optional[str] = None


class DatasetManifest(BaseModel):
    """Top-level manifest stored in .golden_dataset/manifest.json."""

    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    versions: List[DatasetVersion] = Field(default_factory=list)
    current_version: Optional[str] = None


class EvalResult(BaseModel):
    """Per-entry evaluation result."""

    entry_id: str
    question: str
    expected_answer: str
    actual_answer: str
    semantic_similarity: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    overall_score: Optional[float] = None


class EvalSummary(BaseModel):
    """Aggregate evaluation summary for a full dataset."""

    dataset_name: str
    version: str
    evaluated_at: datetime = Field(default_factory=_utcnow)
    total_entries: int
    avg_semantic_similarity: Optional[float] = None
    avg_faithfulness: Optional[float] = None
    avg_answer_relevancy: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_context_precision: Optional[float] = None
    results: List[EvalResult] = Field(default_factory=list)

    def passed(self, threshold: float = 0.5) -> bool:
        if self.avg_semantic_similarity is None:
            return False
        return self.avg_semantic_similarity >= threshold
