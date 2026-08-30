from __future__ import annotations

from pathlib import Path

import pytest

from llm_migrate.service import MigrationService


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def service(project_root: Path) -> MigrationService:
    return MigrationService.from_directory(project_root / "registry")
