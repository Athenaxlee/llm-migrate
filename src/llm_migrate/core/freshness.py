"""Deterministic local registry freshness reporting."""

from __future__ import annotations

from datetime import date

from llm_migrate.core.knowledge import RegistryFreshnessReport, StaleRegistrySection
from llm_migrate.core.models import FreshnessCategory, ModelProfile


def is_stale(profile: ModelProfile, category: FreshnessCategory, as_of_date: date) -> bool:
    """Return whether a recorded category is stale; missing metadata is stale."""
    metadata = profile.freshness.get(category)
    return metadata is None or metadata.is_stale(as_of_date)


def registry_freshness(profiles: list[ModelProfile], as_of_date: date) -> RegistryFreshnessReport:
    sections: list[StaleRegistrySection] = []
    for profile in sorted(profiles, key=lambda item: item.identity.canonical_name):
        for category in FreshnessCategory:
            metadata = profile.freshness.get(category)
            sections.append(
                StaleRegistrySection(
                    canonical_name=profile.identity.canonical_name,
                    category=category,
                    checked_at=metadata.checked_at if metadata else None,
                    max_age_days=metadata.max_age_days if metadata else None,
                    stale=metadata is None or metadata.is_stale(as_of_date),
                )
            )
    return RegistryFreshnessReport(as_of_date=as_of_date, sections=sections)
