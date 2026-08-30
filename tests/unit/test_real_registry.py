from __future__ import annotations

from decimal import Decimal

from llm_migrate.service import MigrationService


def test_real_models_and_migrations_validate(service: MigrationService) -> None:
    report = service.validate_registry()
    assert report.valid
    assert report.model_count == 9
    assert report.migration_count == 3
    assert service.resolve_model("sonnet 5").identity.canonical_name == "claude-sonnet-5"
    assert service.resolve_model("gpt-5.6").identity.canonical_name == "gpt-5.6-sol"


def test_platform_specific_capability_override(service: MigrationService) -> None:
    sonnet = service.resolve_model("sonnet 5")
    assert sonnet.capabilities.structured_output is True
    bedrock = [item for item in sonnet.platforms if item.platform == "amazon-bedrock"]
    assert len(bedrock) == 2
    assert all(item.capability_overrides is not None for item in bedrock)
    assert all(
        item.capability_overrides.structured_output is False
        for item in bedrock
        if item.capability_overrides
    )


def test_reviewed_sonnet_facts_are_promoted_conservatively(
    service: MigrationService,
) -> None:
    sonnet_4_6 = service.resolve_model("sonnet 4.6")
    assert sonnet_4_6.capabilities.maximum_output_tokens == 128000

    sonnet_5 = service.resolve_model("sonnet 5")
    assert len(sonnet_5.evidence_conflicts) == 1
    conflict = sonnet_5.evidence_conflicts[0]
    assert conflict.field_path == (
        "platforms.amazon-bedrock.parameters.adaptive_thinking.disable_supported"
    )
    assert len(conflict.source_ids) == 2


def test_comparison_includes_evidence_backed_migration_knowledge(
    service: MigrationService,
) -> None:
    comparison = service.compare_models("sonnet 4.6", "sonnet 5")
    knowledge = [item for item in comparison.differences if item.category == "migration_knowledge"]
    assert len(knowledge) == 4
    assert all(item.supporting_sources for item in knowledge)
    assert {item.knowledge_id for item in knowledge} == {"claude-sonnet-4-6-to-5"}


def test_resolved_official_pricing_conflict_is_promoted(service: MigrationService) -> None:
    luna = service.resolve_model("gpt 5.6 luna")
    assert luna.evidence_conflicts == []
    assert luna.pricing is not None
    assert luna.pricing.input is not None
    assert luna.pricing.output is not None
    assert luna.pricing.cached_input is not None
    assert luna.pricing.input.amount == Decimal("0.2")
    assert luna.pricing.output.amount == Decimal("1.2")
    assert luna.pricing.cached_input.amount == Decimal("0.02")
