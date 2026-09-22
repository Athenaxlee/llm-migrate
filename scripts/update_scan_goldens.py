"""Regenerate deterministic application-scan golden files.

Scans run through the service (like the golden tests do), so registry-anchored
config-coupling detection sees the reviewed model-id spellings.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.service import MigrationService


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    service = MigrationService.from_directory(root / "registry")
    fixtures = root / "tests" / "fixtures" / "applications"
    output = root / "tests" / "golden" / "application_scans"
    output.mkdir(parents=True, exist_ok=True)
    for application in ("anthropic_app", "openai_app", "bedrock_app", "configured_prompt_app"):
        result = service.scan_application(fixtures / application).model_dump(mode="json")
        result["root"] = "<fixture>"
        (output / f"{application}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
