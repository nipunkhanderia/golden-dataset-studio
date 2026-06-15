"""Dataset store — filesystem-backed versioning for golden datasets.

Layout::

    .golden_dataset/
        manifest.json
        working.jsonl
        versions/
            1.0.jsonl
            1.1.jsonl
        evals/
            eval_1.0_<ts>.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .models import DatasetManifest, DatasetVersion, EvalSummary, GoldenEntry

STUDIO_DIR = ".golden_dataset"


class DatasetStore:
    """Manages golden dataset versioning on the local filesystem."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.studio = self.root / STUDIO_DIR
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._evals_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _versions_dir(self) -> Path:
        return self.studio / "versions"

    @property
    def _evals_dir(self) -> Path:
        return self.studio / "evals"

    @property
    def _manifest_path(self) -> Path:
        return self.studio / "manifest.json"

    @property
    def _working_path(self) -> Path:
        return self.studio / "working.jsonl"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self, name: str, description: str = "") -> DatasetManifest:
        if self._manifest_path.exists():
            raise FileExistsError(
                f"Dataset '{self.load_manifest().name}' already initialised here. "
                "Delete .golden_dataset/ to start fresh."
            )
        manifest = DatasetManifest(name=name, description=description)
        self._save_manifest(manifest)
        return manifest

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _save_manifest(self, manifest: DatasetManifest) -> None:
        self._manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def load_manifest(self) -> DatasetManifest:
        if not self._manifest_path.exists():
            raise FileNotFoundError("No dataset found. Run `golden init <name>` first.")
        return DatasetManifest.model_validate_json(self._manifest_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Working tree
    # ------------------------------------------------------------------

    def get_working_entries(self) -> List[GoldenEntry]:
        if not self._working_path.exists():
            return []
        entries: List[GoldenEntry] = []
        for line in self._working_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(GoldenEntry.model_validate_json(line))
        return entries

    def _save_working_entries(self, entries: List[GoldenEntry]) -> None:
        with open(self._working_path, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(e.model_dump_json() + "\n")

    def add_entry(self, entry: GoldenEntry) -> None:
        with open(self._working_path, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def update_entry(self, entry_id: str, **kwargs) -> GoldenEntry:
        entries = self.get_working_entries()
        matched = False
        for entry in entries:
            if entry.id == entry_id:
                for key, val in kwargs.items():
                    if hasattr(entry, key):
                        object.__setattr__(entry, key, val)
                entry.updated_at = datetime.now(tz=timezone.utc)
                matched = True
                break
        if not matched:
            raise KeyError(f"Entry '{entry_id}' not found in working tree.")
        self._save_working_entries(entries)
        return next(e for e in entries if e.id == entry_id)

    def delete_entry(self, entry_id: str) -> None:
        entries = self.get_working_entries()
        new_entries = [e for e in entries if e.id != entry_id]
        if len(new_entries) == len(entries):
            raise KeyError(f"Entry '{entry_id}' not found in working tree.")
        self._save_working_entries(new_entries)

    # ------------------------------------------------------------------
    # Committing versions
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(entries: List[GoldenEntry]) -> str:
        blob = "\n".join(e.model_dump_json() for e in sorted(entries, key=lambda x: x.id))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _next_version(current: Optional[str]) -> str:
        if current is None:
            return "1.0"
        major, minor = current.split(".")
        return f"{major}.{int(minor) + 1}"

    def commit(self, description: str = "") -> DatasetVersion:
        entries = self.get_working_entries()
        if not entries:
            raise ValueError("Nothing to commit — working tree is empty.")

        manifest = self.load_manifest()
        new_ver = self._next_version(manifest.current_version)
        sha = self._compute_hash(entries)

        vpath = self._versions_dir / f"{new_ver}.jsonl"
        with open(vpath, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(e.model_dump_json() + "\n")

        version = DatasetVersion(
            version=new_ver,
            description=description,
            entry_count=len(entries),
            sha256=sha,
            parent_version=manifest.current_version,
        )
        manifest.versions.append(version)
        manifest.current_version = new_ver
        self._save_manifest(manifest)
        return version

    # ------------------------------------------------------------------
    # Reading versions
    # ------------------------------------------------------------------

    def load_version(self, version: Optional[str] = None) -> List[GoldenEntry]:
        manifest = self.load_manifest()
        ver = version or manifest.current_version
        if not ver:
            raise ValueError("No committed version exists yet. Run `golden commit` first.")
        vpath = self._versions_dir / f"{ver}.jsonl"
        if not vpath.exists():
            raise FileNotFoundError(f"Version {ver} not found.")
        entries: List[GoldenEntry] = []
        for line in vpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(GoldenEntry.model_validate_json(line))
        return entries

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, v1: str, v2: str) -> Dict[str, List[GoldenEntry]]:
        map1 = {e.id: e for e in self.load_version(v1)}
        map2 = {e.id: e for e in self.load_version(v2)}

        added = [map2[k] for k in map2 if k not in map1]
        removed = [map1[k] for k in map1 if k not in map2]
        changed = [
            map2[k] for k in map2 if k in map1 and map1[k].answer != map2[k].answer
        ]
        return {"added": added, "removed": removed, "changed": changed}

    # ------------------------------------------------------------------
    # Evaluation persistence
    # ------------------------------------------------------------------

    def save_eval(self, summary: EvalSummary) -> Path:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self._evals_dir / f"eval_{summary.version}_{ts}.json"
        path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return path

    def list_evals(self) -> List[Path]:
        return sorted(self._evals_dir.glob("eval_*.json"))

    def load_eval(self, path: Path) -> EvalSummary:
        return EvalSummary.model_validate_json(path.read_text(encoding="utf-8"))
