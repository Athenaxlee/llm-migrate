from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.core.models import ModelProfile


def test_checked_in_model_profile_schema_matches_runtime(project_root: Path) -> None:
    checked_in = json.loads(
        (project_root / "registry" / "schemas" / "model-profile.schema.json").read_text()
    )
    assert checked_in == ModelProfile.model_json_schema(mode="validation")
