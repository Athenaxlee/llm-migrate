from __future__ import annotations

from datetime import date

from llm_migrate.core.freshness import is_stale
from llm_migrate.core.models import FreshnessCategory
from llm_migrate.service import MigrationService


def test_fixed_date_freshness_detection(service: MigrationService) -> None:
    profile = service.resolve_model("claude sonnet 5")
    assert not is_stale(profile, FreshnessCategory.PRICING, date(2026, 8, 16))
    assert is_stale(profile, FreshnessCategory.PRICING, date(2026, 8, 17))
    assert not is_stale(profile, FreshnessCategory.CAPABILITIES, date(2026, 9, 8))
    assert is_stale(profile, FreshnessCategory.CAPABILITIES, date(2026, 9, 9))


def test_missing_freshness_is_stale(service: MigrationService) -> None:
    fixture = service.resolve_model("alpha large")
    assert is_stale(fixture, FreshnessCategory.PRICING, date(2026, 8, 8))
