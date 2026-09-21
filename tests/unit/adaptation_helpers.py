"""Shared helpers for adaptation-submission tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from llm_migrate.core.annotations import AnnotatedChange, ChangeEvidence, EvidenceKind
from llm_migrate.core.workspace import GuidanceDisposition
from llm_migrate.service import MigrationService


def dispose_all(
    service: MigrationService,
    run_dir: Path | str,
    source_path: str,
    disposition: Literal["applied", "not_applicable", "declined"] = "applied",
) -> list[GuidanceDisposition]:
    """Dispose every guidance item of one prompt or file task the same way.

    A path without a task has no required guidance, so it needs no
    dispositions.
    """
    tasks = service.list_adaptation_tasks(run_dir)
    prompt_task = next(
        (item for item in tasks.prompt_tasks if item.source_path == source_path), None
    )
    if prompt_task is not None:
        required = [*prompt_task.guidance, *tasks.shared_prompt_guidance]
    else:
        file_task = next(
            (item for item in tasks.file_tasks if item.source_path == source_path), None
        )
        required = list(file_task.required_changes) if file_task is not None else []
    return [
        GuidanceDisposition(guidance_id=item.id, disposition=disposition, note="test disposition")
        for item in required
    ]


def edit_change(
    original_anchor: str,
    adapted_anchor: str,
    why: str = "Adapted for the target model.",
    kind: EvidenceKind = "mechanical",
    url: str = "",
    reference: str = "",
) -> AnnotatedChange:
    """One edit annotation with a single evidence entry."""
    return AnnotatedChange(
        operation="edit",
        original_anchor=original_anchor,
        adapted_anchor=adapted_anchor,
        why=why,
        evidence=[ChangeEvidence(kind=kind, url=url, reference=reference)],
    )


def insert_change(
    adapted_anchor: str,
    why: str = "Added for the target model.",
    kind: EvidenceKind = "mechanical",
    url: str = "",
    reference: str = "",
) -> AnnotatedChange:
    """One insert annotation with a single evidence entry."""
    return AnnotatedChange(
        operation="insert",
        adapted_anchor=adapted_anchor,
        why=why,
        evidence=[ChangeEvidence(kind=kind, url=url, reference=reference)],
    )


def restructure_change(
    why: str = "Restructured for the target model.",
    kind: EvidenceKind = "mechanical",
    url: str = "",
    reference: str = "",
) -> AnnotatedChange:
    """A global restructure annotation claiming the whole rewrite."""
    return AnnotatedChange(
        operation="restructure",
        why=why,
        evidence=[ChangeEvidence(kind=kind, url=url, reference=reference)],
    )
