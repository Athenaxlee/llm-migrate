"""Worklist snapshot with a complete staleness key (V1.5.0-c).

Plan derivation (scan + registry comparison + preparation) dominated the cost
of every submission in the audited run: one full re-derivation per call. The
snapshot persists the derived worklist and its plan evidence in the run
workspace, keyed by a content hash over EVERYTHING the plan derives from —
the scanned application content, `migration.yaml` (retarget/correction
decisions mutate it), the blocker decision log (redesign decisions inject
tasks), the session-registry manifest, the run's observations (they close
unknowns), and the canonical registry content.
Submissions validate against the snapshot and re-derive only when the key
changes. Task *statuses* are never baked in: they are recomputed at read time
from the adaptation log. When in doubt the snapshot is discarded and
re-derived — correctness always wins over the saved scan.

The key also carries the installed toolkit version, and the snapshot schema
version moves whenever derivation output changes shape (v2, v1.6.0-a:
dynamic prompt consumers and the multi-base resolver), so an upgrade never
reuses a worklist derived by older discovery rules.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from llm_migrate.core.blockers import DECISIONS_FILENAME
from llm_migrate.core.models import StrictModel
from llm_migrate.core.observations import OBSERVATIONS_FILENAME
from llm_migrate.core.runstate import atomic_write_text
from llm_migrate.core.unknowns import ContestedFact
from llm_migrate.core.workspace import (
    RUN_CONFIG_FILENAME,
    AdaptationTaskList,
    MigrationRunConfig,
)
from llm_migrate.scanners.python import scannable_files

SNAPSHOT_FILENAME = "worklist-snapshot.yaml"
WORKLIST_MARKER_FILENAME = "worklist-requested"


def mark_worklist_requested(run_dir: Path) -> None:
    """Record that the host explicitly requested the adaptation worklist.

    Only `list_adaptation_tasks` writes this marker. The run-status state
    machine must not infer "moved past research" from the snapshot file:
    every worklist-consulting call (including `get_run_status` itself)
    persists the snapshot as a cache, so its existence proves nothing about
    what the host asked for.
    """
    atomic_write_text(Path(run_dir) / WORKLIST_MARKER_FILENAME, "requested\n")


def worklist_requested(run_dir: Path) -> bool:
    return (Path(run_dir) / WORKLIST_MARKER_FILENAME).is_file()


class WorklistSnapshot(StrictModel):
    """The derived worklist plus plan evidence, valid while `key` matches."""

    schema_version: Literal["2"] = "2"
    run_id: str
    key: str
    tasks: AdaptationTaskList
    evidence_urls: list[str] = Field(default_factory=list)
    # The subset of evidence_urls recorded in the reviewed registry (v1.5.2).
    registry_evidence_urls: list[str] = Field(default_factory=list)
    # Probe deliverables (run-relative) the derivation wrote (v1.6.0-b); a
    # missing one makes the snapshot stale so the next derivation rewrites it.
    probe_paths: list[str] = Field(default_factory=list)
    # Contested registry facts review marks match against (v1.6.0-c).
    contested_facts: list[ContestedFact] = Field(default_factory=list)


def _toolkit_version() -> str:
    try:
        return metadata.version("llm-migrate")
    except metadata.PackageNotFoundError:
        return "unknown"


# (path, size, mtime_ns, ctime_ns) -> sha256. Content is re-read whenever the
# stat signature moves, so the key stays a content hash; the cache only spares
# the long-lived MCP process from re-reading an unchanged application on every
# call. ctime is part of the signature because tools such as `cp -p` and
# `rsync -a` preserve size and mtime while replacing content, and no user tool
# preserves ctime. Bounded so a server that outlives many runs cannot grow
# without limit.
_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}
_DIGEST_CACHE_LIMIT = 50_000


def _file_digest(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return "unreadable"
    signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _DIGEST_CACHE.get(signature)
    if cached is not None:
        return cached
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"
    if len(_DIGEST_CACHE) >= _DIGEST_CACHE_LIMIT:
        _DIGEST_CACHE.clear()
    _DIGEST_CACHE[signature] = digest
    return digest


def snapshot_key(run_dir: Path, config: MigrationRunConfig, registry_digest: str) -> str:
    """Content hash over everything the derived worklist depends on."""
    application = Path(config.application_root)
    digest = hashlib.sha256()
    digest.update(f"registry:{registry_digest}\n".encode())
    digest.update(f"toolkit:{_toolkit_version()}\n".encode())
    base = application if application.is_dir() else application.parent
    if application.exists():
        python_files, candidate_files = scannable_files(application)
        for item in [*python_files, *candidate_files]:
            digest.update(
                f"app:{item.relative_to(base).as_posix()}:{_file_digest(item)}\n".encode()
            )
    else:
        digest.update(b"app:missing\n")
    for name in (
        RUN_CONFIG_FILENAME,
        DECISIONS_FILENAME,
        "session-manifest.yaml",
        OBSERVATIONS_FILENAME,
    ):
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
