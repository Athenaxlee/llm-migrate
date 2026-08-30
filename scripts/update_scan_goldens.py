"""Regenerate deterministic application-scan golden files."""

from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.scanners import scan_application


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests" / "fixtures" / "applications"
    output = root / "tests" / "golden" / "application_scans"
    output.mkdir(parents=True, exist_ok=True)
    for application in ("anthropic_app", "openai_app", "bedrock_app"):
        result = scan_application(fixtures / application).model_dump(mode="json")
        result["root"] = "<fixture>"
        (output / f"{application}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
