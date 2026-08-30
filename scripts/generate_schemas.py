"""Generate checked-in JSON Schemas from authoritative Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.core.models import ModelProfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "registry" / "schemas" / "model-profile.schema.json"
    schema = ModelProfile.model_json_schema(mode="validation")
    target.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
