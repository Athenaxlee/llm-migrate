from __future__ import annotations

import tomllib
from pathlib import Path

import llm_migrate


def test_runtime_version_matches_package_metadata(project_root: Path) -> None:
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert llm_migrate.__version__ == metadata["project"]["version"]
