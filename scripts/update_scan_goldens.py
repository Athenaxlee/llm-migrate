"""Regenerate deterministic application-scan golden files.

Scans run through the service (like the golden tests do), so registry-anchored
config-coupling detection sees the reviewed model-id spellings.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.service import MigrationService

# Golden name -> scan root under tests/fixtures/applications (keep in sync with
# tests/unit/test_application_scanner.py).
SCAN_FIXTURES = {
    "anthropic_app": "anthropic_app",
    "openai_app": "openai_app",
    "bedrock_app": "bedrock_app",
    "configured_prompt_app": "configured_prompt_app",
    "field_pattern_repo": "field_pattern_repo/app",
    "probe_forms_app": "probe_forms_app",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    service = MigrationService.from_directory(root / "registry")
    fixtures = root / "tests" / "fixtures" / "applications"
    output = root / "tests" / "golden" / "application_scans"
    output.mkdir(parents=True, exist_ok=True)
    for application, scan_root in SCAN_FIXTURES.items():
        result = service.scan_application(fixtures / scan_root).model_dump(mode="json")
        result["root"] = "<fixture>"
        (output / f"{application}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
