from __future__ import annotations

from datetime import date

from llm_migrate.core.freshness import is_stale
from llm_migrate.core.models import FreshnessCategory
from llm_migrate.service import MigrationService


def test_fixed_date_freshness_detection(service: MigrationService) -> None:
    profile = service.resolve_model("claude sonnet 5")
    # checked_at 2026-09-24 (maintainer evidence refresh): 7-day pricing window,
    # 30-day capabilities window.
    assert not is_stale(profile, FreshnessCategory.PRICING, date(2026, 10, 1))
    assert is_stale(profile, FreshnessCategory.PRICING, date(2026, 10, 2))
    assert not is_stale(profile, FreshnessCategory.CAPABILITIES, date(2026, 10, 24))
    assert is_stale(profile, FreshnessCategory.CAPABILITIES, date(2026, 10, 25))


def test_missing_freshness_is_stale(service: MigrationService) -> None:
    fixture = service.resolve_model("alpha large")
    assert is_stale(fixture, FreshnessCategory.PRICING, date(2026, 8, 8))
