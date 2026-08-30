from __future__ import annotations

from llm_migrate.core.models import ComparisonSeverity
from llm_migrate.service import MigrationService


def test_comparison_identifies_breaking_capability_and_parameter_changes(
    service: MigrationService,
) -> None:
    result = service.compare_models("alpha large", "gamma cheap")
    categories = {difference.category for difference in result.differences}
    assert "structured_output" in categories
    assert "tool_use" in categories
    assert "parameters" in categories
    assert result.highest_severity is ComparisonSeverity.BREAKING


def test_comparison_is_typed_and_serializable(service: MigrationService) -> None:
    result = service.compare_models("alpha large", "beta balanced")
    dumped = result.model_dump(mode="json")
    assert dumped["source_model"] == "fixture-alpha-large-v1"
    assert isinstance(dumped["differences"], list)
