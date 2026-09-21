"""Shared helpers for adaptation-submission tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from llm_migrate.core.workspace import GuidanceDisposition
from llm_migrate.service import MigrationService


def dispose_all(
    service: MigrationService,
    run_dir: Path | str,
    source_path: str,
    disposition: Literal["applied", "not_applicable", "declined"] = "applied",
) -> list[GuidanceDisposition]:
    """Dispose every guidance item of one prompt task the same way.

    A path without a prompt task has no required guidance, so it needs no
    dispositions.
    """
    tasks = service.list_adaptation_tasks(run_dir)
    task = next((item for item in tasks.prompt_tasks if item.source_path == source_path), None)
    if task is None:
        return []
    return [
        GuidanceDisposition(guidance_id=item.id, disposition=disposition, note="test disposition")
        for item in (*task.guidance, *tasks.shared_prompt_guidance)
    ]
