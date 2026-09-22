"""Worklist snapshot with a complete staleness key (V1.5.0-c).

Plan derivation (scan + registry comparison + preparation) dominated the cost
of every submission in the audited run: one full re-derivation per call. The
snapshot persists the derived worklist and its plan evidence in the run
workspace, keyed by a content hash over EVERYTHING the plan derives from —
the scanned application content, `migration.yaml` (retarget/correction
decisions mutate it), the blocker decision log (redesign decisions inject
tasks), the session-registry manifest, and the canonical registry content.
Submissions validate against the snapshot and re-derive only when the key
changes. Task *statuses* are never baked in: they are recomputed at read time
from the adaptation log. When in doubt the snapshot is discarded and
re-derived — correctness always wins over the saved scan.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from llm_migrate.core.blockers import DECISIONS_FILENAME
from llm_migrate.core.models import StrictModel
from llm_migrate.core.runstate import atomic_write_text
from llm_migrate.core.workspace import (
    RUN_CONFIG_FILENAME,
    AdaptationTaskList,
    MigrationRunConfig,
)
from llm_migrate.scanners.python import scannable_files

SNAPSHOT_FILENAME = "worklist-snapshot.yaml"


class WorklistSnapshot(StrictModel):
    """The derived worklist plus plan evidence, valid while `key` matches."""

    schema_version: Literal["1"] = "1"
    run_id: str
    key: str
    tasks: AdaptationTaskList
    evidence_urls: list[str] = Field(default_factory=list)


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def snapshot_key(run_dir: Path, config: MigrationRunConfig, registry_digest: str) -> str:
    """Content hash over everything the derived worklist depends on."""
    application = Path(config.application_root)
    digest = hashlib.sha256()
    digest.update(f"registry:{registry_digest}\n".encode())
    base = application if application.is_dir() else application.parent
    if application.exists():
        python_files, candidate_files = scannable_files(application)
        for item in [*python_files, *candidate_files]:
            digest.update(
                f"app:{item.relative_to(base).as_posix()}:{_file_digest(item)}\n".encode()
            )
    else:
        digest.update(b"app:missing\n")
    for name in (RUN_CONFIG_FILENAME, DECISIONS_FILENAME, "session-manifest.yaml"):
        digest.update(f"run:{name}:{_file_digest(Path(run_dir) / name)}\n".encode())
    return digest.hexdigest()


def load_snapshot(run_dir: Path, run_id: str) -> WorklistSnapshot | None:
    """The persisted snapshot, or None when absent, foreign, or unreadable.

    An unreadable or foreign snapshot is treated as stale, never trusted:
    correctness wins over the saved derivation.
    """
    path = Path(run_dir) / SNAPSHOT_FILENAME
    if not path.is_file():
        return None
    try:
        snapshot = WorklistSnapshot.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError):
        return None
    if snapshot.run_id != run_id:
        return None
    return snapshot


def save_snapshot(run_dir: Path, snapshot: WorklistSnapshot) -> Path:
    path = Path(run_dir) / SNAPSHOT_FILENAME
    atomic_write_text(path, yaml.safe_dump(snapshot.model_dump(mode="json"), sort_keys=False))
    return path
