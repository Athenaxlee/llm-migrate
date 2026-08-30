from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_migrate.core.models import (
    ComparisonSeverity,
    CompatibilityState,
    ValidationLevel,
)
from llm_migrate.core.registry import RegistryError
from llm_migrate.core.resolver import AmbiguousModelError
from llm_migrate.service import MigrationService

GOLDEN_CASES = {
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


def _fixture(project_root: Path, name: str) -> Path:
    return project_root / "tests" / "fixtures" / "applications" / name


@pytest.mark.parametrize(("name", "case"), GOLDEN_CASES.items())
def test_migration_manifest_matches_golden(
    service: MigrationService,
    project_root: Path,
    name: str,
    case: tuple[str, str, str, str, str],
) -> None:
    application, source, target, source_platform, target_platform = case
    original = Path.cwd()
    try:
        os.chdir(_fixture(project_root, application))
        plan = service.generate_migration_plan(
            Path("."),
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
        )
    finally:
        os.chdir(original)
    actual = service.migration_manifest_as_yaml(plan)
    expected = (project_root / "tests" / "golden" / "v04" / f"{name}.yaml").read_text()
    assert actual == expected


@pytest.mark.parametrize(
    ("application", "source", "target", "source_platform", "target_platform"),
    [
        (
            "anthropic_legacy_app",
            "claude-sonnet-4-5",
            "claude-sonnet-5",
            "anthropic-api",
            "anthropic-api",
        ),
        (
            "anthropic_app",
            "claude-sonnet-5",
            "claude-sonnet-4-6",
            "anthropic-api",
            "amazon-bedrock",
        ),
        (
            "anthropic_app",
            "claude-sonnet-5",
            "gpt-5.6-sol",
            "anthropic-api",
            "openai-api",
        ),
        (
            "openai_app",
            "gpt-5.6-sol",
            "claude-sonnet-5",
            "openai-api",
            "anthropic-api",
        ),
    ],
)
def test_supported_end_to_end_routes_produce_actionable_manifests(
    service: MigrationService,
    project_root: Path,
    application: str,
    source: str,
    target: str,
    source_platform: str,
    target_platform: str,
) -> None:
    root = _fixture(project_root, application)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    plan = service.generate_migration_plan(
        root,
        source,
        target,
        source_platform=source_platform,
        target_platform=target_platform,
    )
    after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert before == after
    assert plan.schema_version == "2"
    assert plan.source.platform == source_platform
    assert plan.target.platform == target_platform
    assert plan.affected_files
    assert plan.required_changes
    assert plan.invocation_changes
    assert plan.required_tests
    assert plan.rollout_recommendations
    assert "No analyzed source files were modified" in service.generate_migration_report(
        root,
        source,
        target,
        source_platform=source_platform,
        target_platform=target_platform,
    )


def test_file_backed_prompt_is_prepared_and_inline_prompt_is_an_unknown(
    service: MigrationService, project_root: Path
) -> None:
    file_backed = service.generate_migration_plan(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    inline = service.generate_migration_plan(
        _fixture(project_root, "anthropic_legacy_app"),
        "claude-sonnet-4-5",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
    )
    assert [item.source_path for item in file_backed.prompt_changes] == ["prompts/system.txt"]
    assert any("Inline or dynamic prompt content" in item for item in inline.unknowns)


def test_invalid_schema_and_context_regression_are_cross_concern_blockers(
    service: MigrationService, project_root: Path
) -> None:
    analysis = service.scan_application(_fixture(project_root, "openai_app"))
    invalid_output = analysis.structured_outputs[0].model_copy(
        update={"json_schema": {"type": "object", "properties": []}}
    )
    requirements = analysis.requirements.model_copy(update={"minimum_context_window": 99_999_999})
    modified = analysis.model_copy(
        update={"structured_outputs": [invalid_output], "requirements": requirements}
    )
    plan = service.generate_migration_plan(
        modified,
        "gpt-5.6-sol",
        "claude-sonnet-5",
        source_platform="openai-api",
        target_platform="anthropic-api",
    )
    codes = {item.code for item in plan.validation_results}
    assert {"invalid_json_schema", "context_window_regression"} <= codes
    assert all(
        item.level is ValidationLevel.BLOCKER
        for item in plan.validation_results
        if item.code in {"invalid_json_schema", "context_window_regression"}
    )
    output_migration = plan.invocation_changes[0].structured_output_migrations[0]
    assert output_migration.state is CompatibilityState.UNKNOWN
    assert output_migration.target_configuration is None
    assert plan.migration_complexity == "blocked"


def test_unknown_ambiguous_model_and_missing_platform_fail_explicitly(
    service: MigrationService, project_root: Path
) -> None:
    application = _fixture(project_root, "anthropic_app")
    with pytest.raises(RegistryError, match="no model matches"):
        service.generate_migration_plan(
            application,
            "does-not-exist",
            "gpt-5.6-sol",
            target_platform="openai-api",
        )
    with pytest.raises(AmbiguousModelError, match="ambiguous model identifier"):
        service.generate_migration_plan(
            application,
            "shared alias",
            "gpt-5.6-sol",
            target_platform="openai-api",
        )
    with pytest.raises(ValueError, match="Source platform is required"):
        service.generate_migration_plan(
            application,
            "sonnet 4.6",
            "gpt-5.6-sol",
            target_platform="openai-api",
        )


def test_unsupported_target_and_incompatible_tools_remain_blockers(
    service: MigrationService, project_root: Path
) -> None:
    plan = service.generate_migration_plan(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "gamma cheap",
        source_platform="anthropic-api",
        target_platform="budget-platform",
    )
    assert plan.overall_migration_risk is ComparisonSeverity.BREAKING
    assert any("tool use" in item for item in plan.blockers)
    assert any("No deterministic invocation adapter" in item for item in plan.blockers)
    assert any("One or both values" in item for item in plan.unknowns)


def test_manifest_affected_files_cover_every_detected_coupling(
    service: MigrationService, project_root: Path
) -> None:
    plan = service.generate_migration_plan(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert plan.affected_files == ["app.py", "prompts/system.txt"]
    categories = {item.category for item in plan.required_changes}
    assert "invocation" in categories
    assert plan.tool_changes
    assert plan.configuration_changes
    assert any(
        location.path == "app.py"
        for change in plan.required_changes
        for location in change.locations
    )


def test_required_tests_only_name_detected_application_concerns(
    service: MigrationService, project_root: Path
) -> None:
    plan = service.generate_migration_plan(
        _fixture(project_root, "anthropic_legacy_app"),
        "claude-sonnet-4-5",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
    )
    joined = " ".join(plan.required_tests)
    assert "prompt intent" in joined
    assert "tool call" not in joined
    assert "streaming" not in joined
    assert "image or document" not in joined
