"""Explainable hard filtering followed by deterministic scoring."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol

from llm_migrate.core.freshness import is_stale
from llm_migrate.core.models import (
    BOOLEAN_CAPABILITY_FIELDS,
    ApplicationRequirements,
    ExcludedCandidate,
    FreshnessCategory,
    LifecycleStatus,
    MigrationGoal,
    ModelProfile,
    ModelRecommendation,
    PlatformAvailability,
    RecommendationConstraints,
    RecommendationResult,
)
from llm_migrate.core.registry import ModelRegistry, RegistryError
from llm_migrate.core.resolver import effective_capabilities

_BOOLEAN_CAPABILITIES = BOOLEAN_CAPABILITY_FIELDS


class ScoringStrategy(Protocol):
    def score(
        self,
        profile: ModelProfile,
        platform: PlatformAvailability,
        constraints: RecommendationConstraints,
    ) -> tuple[Decimal, list[str], list[str]]: ...


class DefaultScoringStrategy:
    def score(
        self,
        profile: ModelProfile,
        platform: PlatformAvailability,
        constraints: RecommendationConstraints,
    ) -> tuple[Decimal, list[str], list[str]]:
        score = Decimal(0)
        reasons: list[str] = []
        tradeoffs: list[str] = []
        capabilities = effective_capabilities(profile, platform)
        if profile.lifecycle.status is LifecycleStatus.ACTIVE:
            score += Decimal(20)
            reasons.append("active lifecycle")
        else:
            tradeoffs.append(f"lifecycle is {profile.lifecycle.status}")
        pricing = profile.pricing
        if constraints.migration_goal is MigrationGoal.LOWER_COST:
            if pricing and pricing.input and pricing.output:
                combined = pricing.input.amount + pricing.output.amount
                score += Decimal(100) / (Decimal(1) + combined)
                reasons.append("lower registry input/output price scores higher")
            else:
                tradeoffs.append("published price is unknown")
        if constraints.migration_goal is MigrationGoal.UPGRADE:
            score += Decimal(capabilities.context_window_tokens or 0) / Decimal(100000)
            reasons.append("larger context window used as a transparent upgrade proxy")
        capability_values = capabilities.model_dump(exclude={"sources"}).values()
        score += Decimal(sum(value is True for value in capability_values))
        reasons.append(f"available on {platform.platform}")
        return score, reasons, tradeoffs


def _candidate_blockers(
    profile: ModelProfile,
    platform: PlatformAvailability,
    constraints: RecommendationConstraints,
    requirements: ApplicationRequirements,
    source: ModelProfile | None,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    unknowns: list[str] = []
    capabilities = effective_capabilities(profile, platform)
    required_capabilities = constraints.required_capabilities | requirements.required_capabilities
    capability_data = capabilities.model_dump()
    for capability in sorted(required_capabilities):
        if capability not in _BOOLEAN_CAPABILITIES:
            raise RegistryError(f"unknown required capability: {capability}")
        value = capability_data[capability]
        if value is False:
            blockers.append(f"required capability {capability!r} is unsupported")
        elif value is None:
            blockers.append(f"required capability {capability!r} is unknown")
            unknowns.append(f"capability {capability!r} has no verified value")
    minimum_context = max(
        value
        for value in (constraints.minimum_context_window, requirements.minimum_context_window, 0)
        if value is not None
    )
    if minimum_context and (capabilities.context_window_tokens or 0) < minimum_context:
        blockers.append(f"context window is below required {minimum_context} tokens")
    for parameter in sorted(requirements.required_parameters):
        support = profile.parameters.get(parameter)
        if support is None:
            blockers.append(f"required parameter {parameter!r} has unknown support")
            unknowns.append(f"parameter {parameter!r} has no registry entry")
        elif support.state.value in {"unsupported", "deprecated"}:
            blockers.append(f"required parameter {parameter!r} is {support.state.value}")
    if constraints.region and constraints.region not in platform.regions:
        blockers.append(f"platform is not verified in region {constraints.region!r}")
    if (
        constraints.migration_goal is MigrationGoal.CROSS_PROVIDER
        and source
        and profile.identity.provider == source.identity.provider
    ):
        blockers.append("candidate does not satisfy the cross-provider goal")
    if (
        source
        and profile.identity.canonical_name == source.identity.canonical_name
        and (
            constraints.source_platform is None
            or platform.platform.casefold() == constraints.source_platform.casefold()
        )
    ):
        blockers.append("candidate is the source model")
    return blockers, unknowns


def recommend_models(
    registry: ModelRegistry,
    constraints: RecommendationConstraints,
    requirements: ApplicationRequirements | None = None,
    strategy: ScoringStrategy | None = None,
    as_of_date: date | None = None,
) -> RecommendationResult:
    scorer = strategy or DefaultScoringStrategy()
    app_requirements = requirements or ApplicationRequirements()
    source = registry.get(constraints.source_model) if constraints.source_model else None
    today = as_of_date or date.today()
    ranked: list[ModelRecommendation] = []
    excluded: list[ExcludedCandidate] = []
    for profile in registry.all():
        profile_blockers: list[str] = []
        if (
            constraints.provider
            and profile.identity.provider.casefold() != constraints.provider.casefold()
        ):
            profile_blockers.append(f"provider is not {constraints.provider!r}")
        platforms = [
            item
            for item in profile.platforms
            if not constraints.platform
            or item.platform.casefold() == constraints.platform.casefold()
        ]
        if not platforms:
            profile_blockers.append(
                f"no {constraints.platform!r} representation"
                if constraints.platform
                else "no platform representation"
            )
        best: ModelRecommendation | None = None
        platform_blockers: list[str] = []
        platform_unknowns: list[str] = []
        for platform in platforms:
            blockers, unknowns = _candidate_blockers(
                profile, platform, constraints, app_requirements, source
            )
            if blockers:
                platform_blockers.extend(f"{platform.platform}: {item}" for item in blockers)
                platform_unknowns.extend(unknowns)
                continue
            score, reasons, tradeoffs = scorer.score(profile, platform, constraints)
            stale = [
                category for category in FreshnessCategory if is_stale(profile, category, today)
            ]
            if stale:
                tradeoffs.append(
                    "stale registry topics: " + ", ".join(item.value for item in stale)
                )
            candidate = ModelRecommendation(
                canonical_name=profile.identity.canonical_name,
                platform=platform.platform,
                score=score,
                reasons=reasons,
                tradeoffs=tradeoffs,
                unknowns=unknowns,
                stale_categories=stale,
                evidence_state=(
                    "fixture"
                    if profile.fixture
                    else "conflicting"
                    if profile.evidence_conflicts
                    else "verified"
                ),
            )
            if best is None or candidate.score > best.score:
                best = candidate
        if best is not None and not profile_blockers:
            ranked.append(best)
        else:
            excluded.append(
                ExcludedCandidate(
                    canonical_name=profile.identity.canonical_name,
                    blockers=sorted(set([*profile_blockers, *platform_blockers])),
                    unknowns=sorted(set(platform_unknowns)),
                )
            )
    ranked.sort(key=lambda item: (-item.score, item.canonical_name, item.platform or ""))
    excluded.sort(key=lambda item: item.canonical_name)
    return RecommendationResult(
        constraints=constraints,
        application_requirements=requirements,
        recommendations=ranked,
        excluded_candidates=excluded,
    )
