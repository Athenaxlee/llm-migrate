from __future__ import annotations

import pytest

from llm_migrate.core.models import MigrationGoal, RecommendationConstraints
from llm_migrate.core.registry import RegistryError
from llm_migrate.service import MigrationService


def test_hard_capability_filter_excludes_cheaper_incompatible_model(
    service: MigrationService,
) -> None:
    result = service.recommend_models(
        RecommendationConstraints(
            platform="fixture-api",
            required_capabilities={"tool_use"},
            migration_goal=MigrationGoal.LOWER_COST,
        )
    )
    assert [item.canonical_name for item in result.recommendations] == [
        "fixture-beta-balanced-v2",
        "fixture-alpha-large-v1",
    ]


def test_recommendation_order_is_deterministic(service: MigrationService) -> None:
    constraints = RecommendationConstraints(migration_goal=MigrationGoal.LOWER_COST)
    first = service.recommend_models(constraints)
    second = service.recommend_models(constraints)
    assert first == second
    assert first.recommendations[0].canonical_name == "fixture-gamma-cheap-v1"


def test_cross_provider_uses_resolved_source(service: MigrationService) -> None:
    result = service.recommend_models(
        RecommendationConstraints(
            source_model="alpha large",
            provider="example-ai",
            migration_goal=MigrationGoal.CROSS_PROVIDER,
        )
    )
    assert {item.canonical_name for item in result.recommendations} == {
        "fixture-beta-balanced-v2",
        "fixture-gamma-cheap-v1",
    }


def test_unknown_capability_is_actionable(service: MigrationService) -> None:
    with pytest.raises(RegistryError, match="unknown required capability"):
        service.recommend_models(RecommendationConstraints(required_capabilities={"telepathy"}))
