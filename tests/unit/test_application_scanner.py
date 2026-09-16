from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_migrate.core.models import CouplingKind, RecommendationConstraints
from llm_migrate.service import MigrationService


@pytest.mark.parametrize(
    "application", ["anthropic_app", "openai_app", "bedrock_app", "configured_prompt_app"]
)
def test_scans_match_golden_outputs(
    service: MigrationService, project_root: Path, application: str
) -> None:
    root = project_root / "tests" / "fixtures" / "applications" / application
    actual = service.scan_application(root).model_dump(mode="json")
    actual["root"] = "<fixture>"
    expected = json.loads(
        (
            project_root / "tests" / "golden" / "application_scans" / f"{application}.json"
        ).read_text()
    )
    assert actual == expected


def test_scan_requirements_drive_recommendation(
    service: MigrationService, project_root: Path
) -> None:
    analysis = service.scan_application(
        project_root / "tests" / "fixtures" / "applications" / "anthropic_app"
    )
    result = service.recommend_models(RecommendationConstraints(), analysis)
    assert "fixture-alpha-large-v1" in {item.canonical_name for item in result.recommendations}
    assert result.constraints.source_model == "claude-sonnet-5"
    assert "claude-sonnet-5" not in {item.canonical_name for item in result.recommendations}
    gamma = next(
        item
        for item in result.excluded_candidates
        if item.canonical_name == "fixture-gamma-cheap-v1"
    )
    assert any("tool_use" in blocker for blocker in gamma.blockers)


def test_scanner_preserves_source_locations(service: MigrationService, project_root: Path) -> None:
    analysis = service.scan_application(
        project_root / "tests" / "fixtures" / "applications" / "openai_app"
    )
    model = next(item for item in analysis.findings if item.kind is CouplingKind.MODEL_IDENTIFIER)
    assert model.location.path == "app.py"
    assert model.location.line == 8
    assert "image_input" in analysis.requirements.required_capabilities


def test_scan_resolve_compare_integration(service: MigrationService, project_root: Path) -> None:
    analysis = service.scan_application(
        project_root / "tests" / "fixtures" / "applications" / "anthropic_app"
    )
    assert any(item.value == "prompts/system.txt" for item in analysis.findings)
    source_identifier = next(iter(analysis.requirements.source_models))
    source = service.resolve_model(source_identifier)
    comparison = service.compare_models(source.canonical_name, "gpt-5.6-terra")
    assert comparison.source_model == "claude-sonnet-5"
    assert comparison.target_model == "gpt-5.6-terra"


def test_platform_migration_can_recommend_same_model_identity(
    service: MigrationService, project_root: Path
) -> None:
    analysis = service.scan_application(
        project_root / "tests" / "fixtures" / "applications" / "bedrock_app"
    )
    result = service.recommend_models(RecommendationConstraints(platform="anthropic-api"), analysis)
    assert result.constraints.source_platform == "amazon-bedrock"
    assert "claude-sonnet-4-6" in {item.canonical_name for item in result.recommendations}
