"""Regenerate deterministic V0.4 migration-manifest golden files."""

from __future__ import annotations

import os
from pathlib import Path

from llm_migrate.service import MigrationService

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "anthropic_upgrade": (
        "anthropic_legacy_app",
        "claude-sonnet-4-5",
        "claude-sonnet-5",
        "anthropic-api",
        "anthropic-api",
    ),
    "anthropic_to_bedrock": (
        "anthropic_app",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "anthropic-api",
        "amazon-bedrock",
    ),
    "anthropic_to_openai": (
        "anthropic_app",
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "anthropic-api",
        "openai-api",
    ),
    "openai_to_anthropic": (
        "openai_app",
        "gpt-5.6-sol",
        "claude-sonnet-5",
        "openai-api",
        "anthropic-api",
    ),
}


def main() -> None:
    service = MigrationService.from_directory(ROOT / "registry")
    output = ROOT / "tests" / "golden" / "v04"
    output.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    try:
        for name, (application, source, target, source_platform, target_platform) in CASES.items():
            os.chdir(ROOT / "tests" / "fixtures" / "applications" / application)
            plan = service.generate_migration_plan(
                Path("."),
                source,
                target,
                source_platform=source_platform,
                target_platform=target_platform,
            )
            (output / f"{name}.yaml").write_text(
                service.migration_manifest_as_yaml(plan), encoding="utf-8"
            )
    finally:
        os.chdir(original)


if __name__ == "__main__":
    main()
