"""Importers — parse external files into GoldenEntry lists.

Supported: JSONL, CSV, JSON with field alias auto-detection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .models import GoldenEntry


def _coerce_entry(data: Dict[str, Any]) -> GoldenEntry:
    question = (
        data.get("question") or data.get("q") or data.get("prompt") or data.get("input") or ""
    )
    answer = (
        data.get("answer") or data.get("a") or data.get("output")
        or data.get("expected") or data.get("ground_truth") or ""
    )
    raw_contexts = data.get("contexts") or data.get("context") or data.get("retrieved_contexts") or []
    if isinstance(raw_contexts, str):
        raw_contexts = [raw_contexts] if raw_contexts.strip() else []

    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    metadata = data.get("metadata") or data.get("meta") or {}

    return GoldenEntry(
        question=question,
        answer=answer,
        contexts=raw_contexts,
        tags=tags,
        metadata=metadata,
    )


def import_jsonl(path: Path) -> List[GoldenEntry]:
    entries: List[GoldenEntry] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
        entries.append(_coerce_entry(data))
    return entries


def import_csv(path: Path) -> List[GoldenEntry]:
    entries: List[GoldenEntry] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_ctx = row.get("contexts", row.get("context", ""))
            contexts = [c.strip() for c in raw_ctx.split("|") if c.strip()]

            raw_tags = row.get("tags", "")
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]

            raw_meta = row.get("metadata", row.get("meta", "{}"))
            try:
                metadata = json.loads(raw_meta) if raw_meta.strip() else {}
            except json.JSONDecodeError:
                metadata = {}

            entries.append(
                GoldenEntry(
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    contexts=contexts,
                    tags=tags,
                    metadata=metadata,
                )
            )
    return entries


def import_json(path: Path) -> List[GoldenEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a top-level JSON array.")
    return [_coerce_entry(item) for item in raw]
