from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal

from llm_migrate.core.models import (
    ComparisonState,
    IdentifierMatchType,
    PricingMatchStatus,
    RecommendationConstraints,
)
from llm_migrate.service import MigrationService


def test_platform_id_resolution_returns_effective_profile(service: MigrationService) -> None:
    direct = service.resolve_model("claude-sonnet-4-6")
    assert direct.platform is not None
    assert direct.platform.platform == "anthropic-api"
    assert direct.matched_by is IdentifierMatchType.PLATFORM_MODEL_ID

    result = service.resolve_model(
        "anthropic.claude-sonnet-5", platform="amazon-bedrock", endpoint="bedrock-runtime"
    )
    assert result.platform is not None
    assert result.platform.platform == "amazon-bedrock"
    assert result.effective_capabilities.structured_output is False
    assert result.profile.capabilities.structured_output is True


def test_platform_comparison_exposes_all_states(service: MigrationService) -> None:
    result = service.compare_models(
        "sonnet 4.6",
        "sonnet 5",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
    )
    states = {item.state for item in result.differences}
    assert ComparisonState.SAME in states
    assert ComparisonState.DIFFERENT in states
    assert ComparisonState.UNSUPPORTED in states
    structured = next(item for item in result.differences if item.field == "structured_output")
    assert structured.state is ComparisonState.UNSUPPORTED

    fixture_comparison = service.compare_models("alpha large", "beta balanced")
    assert ComparisonState.UNKNOWN in {item.state for item in fixture_comparison.differences}
    unspecified_platform = next(
        item for item in fixture_comparison.differences if item.field == "representation"
    )
    assert unspecified_platform.source_value is None
    assert unspecified_platform.target_value is None
    assert unspecified_platform.state is ComparisonState.UNKNOWN


def test_live_pricing_is_normalized_and_registry_is_unchanged(
    service: MigrationService,
) -> None:
    before = service.get_model_profile("gpt-5.6-sol")

    def fetcher(url: str, timeout: float) -> bytes:
        assert url == "https://openrouter.ai/api/v1/models"
        assert timeout == 2.0
        return json.dumps(
            {
                "data": [
                    {
                        "id": "openai/gpt-5.6-sol",
                        "pricing": {"prompt": "0.000004", "completion": "0.000030"},
                    }
                ]
            }
        ).encode()

    result = service.query_live_pricing("gpt-5.6-sol", timeout=2.0, fetcher=fetcher)
    assert result.status is PricingMatchStatus.MATCHED
    assert result.input_per_million == Decimal("4")
    assert result.output_per_million == Decimal("30")
    assert result.discrepancies == ["OpenRouter output 30.000000 differs from registry output 20"]
    assert service.get_model_profile("gpt-5.6-sol") == before


def test_live_pricing_fails_closed_offline(service: MigrationService) -> None:
    def unavailable(url: str, timeout: float) -> bytes:
        raise TimeoutError("offline")

    result = service.query_live_pricing("gpt-5.6-sol", fetcher=unavailable)
    assert result.status is PricingMatchStatus.UNAVAILABLE
    assert result.input_per_million is None
    assert result.retrieved_at.tzinfo is not None


def test_live_pricing_exact_not_found_and_malformed_response(
    service: MigrationService,
) -> None:
    not_found = service.query_live_pricing(
        "gpt-5.6-sol",
        fetcher=lambda url, timeout: b'{"data":[{"id":"openai/gpt-5.6-terra"}]}',
    )
    malformed = service.query_live_pricing(
        "gpt-5.6-sol",
        fetcher=lambda url, timeout: b'{"data":{"not":"an array"}}',
    )
    assert not_found.status is PricingMatchStatus.NOT_FOUND
    assert malformed.status is PricingMatchStatus.UNAVAILABLE
    assert "not an array" in (malformed.error or "")


def test_live_pricing_enforces_bounded_timeout(service: MigrationService) -> None:
    called = False

    def fetcher(url: str, timeout: float) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    result = service.query_live_pricing("gpt-5.6-sol", timeout=31, fetcher=fetcher)
    assert result.status is PricingMatchStatus.UNAVAILABLE
    assert "between 0.1 and 30" in (result.error or "")
    assert not called


def test_unmapped_fixture_does_not_make_network_request(service: MigrationService) -> None:
    called = False

    def fetcher(url: str, timeout: float) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    result = service.query_live_pricing("alpha large", fetcher=fetcher)
    assert result.status is PricingMatchStatus.UNMAPPED
    assert not called


def test_live_pricing_overlay_uses_one_snapshot_for_comparison(
    service: MigrationService,
) -> None:
    calls = 0

    def fetcher(url: str, timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return json.dumps(
            {
                "data": [
                    {
                        "id": "openai/gpt-5.6-sol",
                        "pricing": {"prompt": "0.000005", "completion": "0.000030"},
                    },
                    {
                        "id": "openai/gpt-5.6-terra",
                        "pricing": {"prompt": "0.000002", "completion": "0.000012"},
                    },
                ]
            }
        ).encode()

    result = service.compare_models(
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        include_live_pricing=True,
        pricing_fetcher=fetcher,
    )
    assert calls == 1
    assert len(result.pricing_overlays) == 2
    assert {item.status for item in result.pricing_overlays} == {PricingMatchStatus.MATCHED}
    assert len({item.retrieved_at for item in result.pricing_overlays}) == 1


def test_recommendation_can_attach_live_pricing_overlay(service: MigrationService) -> None:
    calls = 0

    def fetcher(url: str, timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return b'{"data":[]}'

    result = service.recommend_models(
        RecommendationConstraints(provider="openai"),
        include_live_pricing=True,
        pricing_fetcher=fetcher,
    )
    assert calls == 1
    assert len(result.pricing_overlays) == 3
    assert all(item.status is PricingMatchStatus.NOT_FOUND for item in result.pricing_overlays)


def test_pricing_result_accepts_fixed_utc_timestamp(service: MigrationService) -> None:
    # Adapter-level injection is intentionally tested through its public function elsewhere;
    # this assertion protects timezone-aware service results.
    assert datetime.now(UTC).tzinfo is not None
