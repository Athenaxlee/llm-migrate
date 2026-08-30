from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.core.models import CompatibilityState
from llm_migrate.service import MigrationService


def _fixture(project_root: Path, name: str) -> Path:
    return project_root / "tests" / "fixtures" / "applications" / name


def test_anthropic_to_openai_matches_review_surface_golden(
    service: MigrationService, project_root: Path
) -> None:
    result = service.prepare_invocation_migration(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    review_surface = {
        "target": result.target.model_dump(mode="json"),
        "parameter_mappings": [item.model_dump(mode="json") for item in result.parameter_mappings],
        "required_changes": result.required_changes,
        "warnings": result.warnings,
        "blockers": result.blockers,
        "tool_compatibility": result.source_analysis.tool_compatibility.model_dump(mode="json")
        if result.source_analysis.tool_compatibility
        else None,
    }
    expected = json.loads(
        (project_root / "tests" / "golden" / "v03" / "anthropic_to_openai.json").read_text()
    )
    assert review_surface == expected


def test_direct_anthropic_to_bedrock_and_bedrock_to_direct(
    service: MigrationService, project_root: Path
) -> None:
    direct_to_bedrock = service.prepare_invocation_migration(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",
    )
    assert direct_to_bedrock.target.operation == "converse"
    assert direct_to_bedrock.target.model_field == "modelId"
    assert direct_to_bedrock.parameter_mappings[0].target_name == "maxTokens"
    assert direct_to_bedrock.parameter_mappings[0].target_container == "inferenceConfig"
    assert any("authentication" in item for item in direct_to_bedrock.required_changes)
    bedrock_tool = direct_to_bedrock.tool_schema_migrations[0]
    assert bedrock_tool.state is CompatibilityState.MECHANICALLY_CONVERTIBLE
    assert bedrock_tool.target_field == "toolConfig.tools"
    assert bedrock_tool.target_definition == {
        "toolSpec": {
            "name": "get_weather",
            "description": "Return current weather for a city.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
        }
    }
    assert direct_to_bedrock.configuration_migration is not None
    assert direct_to_bedrock.configuration_migration.state is (
        CompatibilityState.MECHANICALLY_CONVERTIBLE
    )
    assert direct_to_bedrock.configuration_migration.credential_strategy == "aws_default_chain"
    assert direct_to_bedrock.configuration_migration.region_environment_variables == [
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ]

    bedrock_to_direct = service.prepare_invocation_migration(
        _fixture(project_root, "bedrock_app"),
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="amazon-bedrock",
        target_platform="anthropic-api",
    )
    assert bedrock_to_direct.target.operation == "messages.create"
    assert bedrock_to_direct.source_analysis.multimodal_inputs == ["document_input"]
    assert bedrock_to_direct.source_analysis.multimodal_contracts[0].provider_format == (
        "messages.content.document"
    )
    assert bedrock_to_direct.source_analysis.multimodal_contracts[0].source_kind == "bytes"
    assert any("document_input payload" in item for item in bedrock_to_direct.required_changes)
    assert any(
        "Add required target parameter max_tokens" in item
        for item in bedrock_to_direct.required_changes
    )
    assert any(
        "Target requires parameter max_tokens" in item for item in bedrock_to_direct.warnings
    )
    assert not bedrock_to_direct.blockers


def test_unsupported_tool_behavior_is_a_blocker_and_files_are_unchanged(
    service: MigrationService, project_root: Path
) -> None:
    application = _fixture(project_root, "anthropic_app")
    before = {path: path.read_bytes() for path in application.rglob("*") if path.is_file()}
    result = service.prepare_invocation_migration(
        application,
        "claude-sonnet-5",
        "gamma cheap",
        source_platform="anthropic-api",
        target_platform="budget-platform",
    )
    after = {path: path.read_bytes() for path in application.rglob("*") if path.is_file()}
    assert before == after
    assert result.source_analysis.tool_compatibility is not None
    assert result.source_analysis.tool_compatibility.state is CompatibilityState.UNSUPPORTED
    assert any("tool use" in blocker for blocker in result.blockers)
    assert any("No deterministic invocation adapter" in blocker for blocker in result.blockers)
    assert result.tool_schema_migrations[0].state is CompatibilityState.UNSUPPORTED
    assert result.tool_schema_migrations[0].target_definition is None


def test_openai_structured_output_compatibility_is_explicit(
    service: MigrationService, project_root: Path
) -> None:
    result = service.analyze_invocation(
        _fixture(project_root, "openai_app"),
        target="claude-sonnet-5",
        target_platform="anthropic-api",
    )
    assert result.structured_output_compatibility is not None
    assert (
        result.structured_output_compatibility.state is CompatibilityState.MECHANICALLY_CONVERTIBLE
    )
    assert result.reasoning_controls == ["reasoning_effort"]
    assert result.structured_output_fields == ["text"]
    assert result.response_parsers == ["json.loads"]
    assert result.parser_assumptions[0].strictness == "syntax"
    assert result.structured_outputs[0].json_schema is not None
    assert result.multimodal_contracts[0].provider_format == "input.content.input_image"
    assert result.multimodal_contracts[0].source_kind == "url"


def test_tool_and_output_migrations_emit_reviewable_target_payloads(
    service: MigrationService, project_root: Path
) -> None:
    anthropic_to_openai = service.prepare_invocation_migration(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    openai_tool = anthropic_to_openai.tool_schema_migrations[0]
    assert openai_tool.state is CompatibilityState.MECHANICALLY_CONVERTIBLE
    assert openai_tool.target_field == "tools"
    assert openai_tool.target_definition is not None
    assert openai_tool.target_definition["type"] == "function"
    assert openai_tool.target_definition["parameters"]["required"] == ["city"]
    assert anthropic_to_openai.configuration_migration is not None
    assert anthropic_to_openai.configuration_migration.credential_environment_variables == [
        "OPENAI_API_KEY"
    ]

    openai_to_anthropic = service.prepare_invocation_migration(
        _fixture(project_root, "openai_app"),
        "gpt-5.6-sol",
        "claude-sonnet-5",
        source_platform="openai-api",
        target_platform="anthropic-api",
    )
    anthropic_tool = openai_to_anthropic.tool_schema_migrations[0]
    output = openai_to_anthropic.structured_output_migrations[0]
    assert anthropic_tool.target_definition is not None
    assert "input_schema" in anthropic_tool.target_definition
    assert output.state is CompatibilityState.MECHANICALLY_CONVERTIBLE
    assert output.target_field == "output_config"
    assert output.target_configuration is not None
    assert output.target_configuration["format"]["type"] == "json_schema"
    assert output.target_configuration["format"]["schema"]["required"] == ["vendor_id"]

    openai_to_bedrock = service.prepare_invocation_migration(
        _fixture(project_root, "openai_app"),
        "gpt-5.6-sol",
        "claude-sonnet-4-6",
        source_platform="openai-api",
        target_platform="amazon-bedrock",
    )
    unsupported_output = openai_to_bedrock.structured_output_migrations[0]
    assert unsupported_output.state is CompatibilityState.UNSUPPORTED
    assert unsupported_output.target_configuration is None
    assert unsupported_output.review_notes


def test_source_identity_mismatch_is_a_blocker(
    service: MigrationService, project_root: Path
) -> None:
    result = service.prepare_invocation_migration(
        _fixture(project_root, "openai_app"),
        "claude-sonnet-5",
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert any("source model does not match" in item for item in result.blockers)
    assert any("source provider does not match" in item for item in result.blockers)
    assert any("source platform does not match" in item for item in result.blockers)


def test_unsupported_and_reasoning_parameters_are_not_false_direct_mappings(
    service: MigrationService, project_root: Path
) -> None:
    unsupported = service.prepare_invocation_migration(
        _fixture(project_root, "anthropic_app"),
        "claude-sonnet-5",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
    )
    temperature = next(
        item for item in unsupported.parameter_mappings if item.source_name == "temperature"
    )
    assert unsupported.configuration_migration is not None
    assert unsupported.configuration_migration.state is CompatibilityState.DIRECTLY_COMPATIBLE
    assert temperature.state is CompatibilityState.UNSUPPORTED
    assert temperature.target_name is None
    assert any("temperature" in item for item in unsupported.blockers)

    reasoning = service.prepare_invocation_migration(
        _fixture(project_root, "openai_app"),
        "gpt-5.6-sol",
        "claude-sonnet-4-6",
        source_platform="openai-api",
        target_platform="anthropic-api",
    )
    effort = next(
        item for item in reasoning.parameter_mappings if item.source_name == "reasoning_effort"
    )
    assert effort.target_name == "effort"
    assert effort.state is CompatibilityState.SEMANTICALLY_DIFFERENT
    assert any("reasoning-control semantics" in item for item in reasoning.warnings)


def test_tool_compatibility_exercises_all_contract_states(
    service: MigrationService, project_root: Path
) -> None:
    analysis = service.scan_application(_fixture(project_root, "anthropic_app"))
    direct = service.analyze_invocation(
        analysis, target="claude-sonnet-5", target_platform="anthropic-api"
    )
    mechanical = service.analyze_invocation(
        analysis, target="gpt-5.6-sol", target_platform="openai-api"
    )
    advanced_schema_analysis = analysis.model_copy(
        update={
            "tool_definitions": [
                analysis.tool_definitions[0].model_copy(
                    update={
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "city": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                            },
                        }
                    }
                )
            ]
        }
    )
    semantic = service.analyze_invocation(
        advanced_schema_analysis, target="gpt-5.6-sol", target_platform="openai-api"
    )
    unknown = service.analyze_invocation(
        analysis.model_copy(update={"tool_definitions": []}),
        target="gpt-5.6-sol",
        target_platform="openai-api",
    )
    unsupported = service.analyze_invocation(
        analysis, target="gamma cheap", target_platform="budget-platform"
    )
    assert direct.tool_compatibility is not None
    assert mechanical.tool_compatibility is not None
    assert semantic.tool_compatibility is not None
    assert unknown.tool_compatibility is not None
    assert unsupported.tool_compatibility is not None
    assert direct.tool_compatibility.state is CompatibilityState.DIRECTLY_COMPATIBLE
    assert mechanical.tool_compatibility.state is CompatibilityState.MECHANICALLY_CONVERTIBLE
    assert semantic.tool_compatibility.state is CompatibilityState.SEMANTICALLY_DIFFERENT
    assert any("anyOf" in item for item in semantic.tool_compatibility.issues)
    assert unknown.tool_compatibility.state is CompatibilityState.UNKNOWN
    assert unsupported.tool_compatibility.state is CompatibilityState.UNSUPPORTED


def test_structured_output_schema_strictness_and_parser_are_analyzed(
    service: MigrationService, project_root: Path
) -> None:
    analysis = service.scan_application(_fixture(project_root, "openai_app"))
    direct = service.analyze_invocation(
        analysis, target="gpt-5.6-sol", target_platform="openai-api"
    )
    mechanical = service.analyze_invocation(
        analysis, target="claude-sonnet-5", target_platform="anthropic-api"
    )
    strict_analysis = analysis.model_copy(
        update={
            "structured_outputs": [
                analysis.structured_outputs[0].model_copy(update={"strict": True})
            ]
        }
    )
    semantic = service.analyze_invocation(
        strict_analysis, target="claude-sonnet-5", target_platform="anthropic-api"
    )
    unresolved_analysis = analysis.model_copy(
        update={
            "structured_outputs": [
                analysis.structured_outputs[0].model_copy(update={"json_schema": None})
            ]
        }
    )
    unknown = service.analyze_invocation(
        unresolved_analysis, target="claude-sonnet-5", target_platform="anthropic-api"
    )
    unsupported = service.analyze_invocation(
        analysis, target="gamma cheap", target_platform="budget-platform"
    )
    assessments = [
        direct.structured_output_compatibility,
        mechanical.structured_output_compatibility,
        semantic.structured_output_compatibility,
        unknown.structured_output_compatibility,
        unsupported.structured_output_compatibility,
    ]
    assert all(item is not None for item in assessments)
    assert [item.state for item in assessments if item is not None] == [
        CompatibilityState.DIRECTLY_COMPATIBLE,
        CompatibilityState.MECHANICALLY_CONVERTIBLE,
        CompatibilityState.SEMANTICALLY_DIFFERENT,
        CompatibilityState.UNKNOWN,
        CompatibilityState.UNSUPPORTED,
    ]

    bedrock_preparation = service.prepare_invocation_migration(
        analysis,
        "gpt-5.6-sol",
        "claude-sonnet-4-6",
        source_platform="openai-api",
        target_platform="amazon-bedrock",
    )
    assert any(
        "No deterministic structured-output configuration mapping" in item
        for item in bedrock_preparation.blockers
    )
