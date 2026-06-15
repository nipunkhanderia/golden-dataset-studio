"""Exporters — serialise GoldenEntry lists to standard formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

from .models import GoldenEntry


def export_jsonl(entries: List[GoldenEntry], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(entry.model_dump_json() + "\n")


def export_csv(entries: List[GoldenEntry], path: Path) -> None:
    fieldnames = ["id", "question", "answer", "contexts", "tags", "metadata"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow(
                {
                    "id": e.id,
                    "question": e.question,
                    "answer": e.answer,
                    "contexts": " | ".join(e.contexts),
                    "tags": ", ".join(e.tags),
                    "metadata": json.dumps(e.metadata),
                }
            )


def export_json(entries: List[GoldenEntry], path: Path) -> None:
    data = [json.loads(e.model_dump_json()) for e in entries]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
