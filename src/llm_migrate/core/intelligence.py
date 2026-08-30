"""Point-in-time lifecycle and canonical-price model intelligence."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from llm_migrate.core.models import (
    EndpointCostEstimate,
    FreshnessCategory,
    LifecycleStatus,
    MigrationCostEstimate,
    MigrationWorkload,
    ModelLifecycleCheck,
    ModelProfile,
    SourceReference,
)

_MILLION = Decimal(1_000_000)


def check_model_lifecycle(
    profile: ModelProfile,
    *,
    as_of_date: date,
) -> ModelLifecycleCheck:
    """Interpret reviewed lifecycle facts as of a caller-supplied date."""
    lifecycle = profile.lifecycle
    end_dates = [
        value
        for value in (lifecycle.earliest_end_of_life_on, lifecycle.end_of_life_on)
        if value is not None
    ]
    end_date = min(end_dates) if end_dates else None
    days_until = (end_date - as_of_date).days if end_date is not None else None
    freshness = profile.freshness.get(FreshnessCategory.LIFECYCLE)
    stale = freshness is None or freshness.is_stale(as_of_date)
    warnings: list[str] = []
    actions: list[str] = []

    expired = days_until is not None and days_until <= 0
    deprecated = (
        lifecycle.status is LifecycleStatus.DEPRECATED
        or lifecycle.deprecated_on is not None
        and lifecycle.deprecated_on <= as_of_date
    )
    unsuitable_status = (
        lifecycle.status
        in {
            LifecycleStatus.LEGACY,
            LifecycleStatus.END_OF_LIFE,
        }
        or deprecated
    )
    suitable = not expired and not unsuitable_status

    if expired:
        warnings.append(f"Recorded end-of-life date {end_date} has passed.")
        actions.append("Do not select this model for a new migration; choose an active successor.")
    elif deprecated:
        warnings.append(
            "The model is deprecated as of "
            f"{lifecycle.deprecated_on or 'the reviewed registry status'}."
        )
        actions.append(
            "Plan migration to an active successor before the recorded end-of-life date."
        )
    elif lifecycle.status is LifecycleStatus.LEGACY:
        warnings.append("The registry marks this model as legacy.")
        actions.append("Prefer an active model for new deployments and validate a successor.")
    elif days_until is not None and days_until <= 90:
        warnings.append(f"The earliest recorded end-of-life date is {days_until} days away.")
        actions.append("Complete successor evaluation before the recorded end-of-life date.")
    if stale:
        warnings.append("Lifecycle evidence is stale or has no freshness record.")
        actions.append("Review current official lifecycle documentation before production rollout.")

    return ModelLifecycleCheck(
        model=profile.identity.canonical_name,
        status=lifecycle.status,
        as_of_date=as_of_date,
        suitable_for_new_migrations=suitable,
        days_until_end_of_life=days_until,
        lifecycle_facts_stale=stale,
        warnings=warnings,
        recommended_actions=actions,
        sources=lifecycle.sources or profile.sources,
    )


def _endpoint_cost(profile: ModelProfile, workload: MigrationWorkload) -> EndpointCostEstimate:
    pricing = profile.pricing
    input_price = pricing.input if pricing is not None else None
    output_price = pricing.output if pricing is not None else None
    input_cost = (
        Decimal(workload.requests * workload.input_tokens_per_request)
        * input_price.amount
        / _MILLION
        if input_price is not None
        else None
    )
    output_cost = (
        Decimal(workload.requests * workload.output_tokens_per_request)
        * output_price.amount
        / _MILLION
        if output_price is not None
        else None
    )
    unknown: list[Literal["input", "output"]] = []
    if input_cost is None:
        unknown.append("input")
    if output_cost is None:
        unknown.append("output")
    sources_by_id: dict[str, SourceReference] = {}
    for price in (input_price, output_price):
        if price is not None:
            sources_by_id.update({source.id: source for source in price.sources})
    if not sources_by_id:
        sources_by_id.update(
            {
                source.id: source
                for source in profile.sources
                if any(
                    support == "pricing" or support.startswith("pricing.")
                    for support in source.supports
                )
            }
        )
    return EndpointCostEstimate(
        model=profile.identity.canonical_name,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=(
            input_cost + output_cost if input_cost is not None and output_cost is not None else None
        ),
        unknown_price_components=unknown,
        sources=list(sources_by_id.values()),
    )


def estimate_migration_cost(
    source: ModelProfile,
    target: ModelProfile,
    workload: MigrationWorkload,
) -> MigrationCostEstimate:
    """Estimate recurring canonical token cost; never query or refresh pricing."""
    source_cost = _endpoint_cost(source, workload)
    target_cost = _endpoint_cost(target, workload)
    delta: Decimal | None = None
    savings: Decimal | None = None
    if source_cost.total_cost_usd is not None and target_cost.total_cost_usd is not None:
        delta = target_cost.total_cost_usd - source_cost.total_cost_usd
        if source_cost.total_cost_usd != 0:
            savings = (-delta / source_cost.total_cost_usd * Decimal(100)).quantize(Decimal("0.01"))
    caveats = [
        "Estimate uses checked-in canonical per-token pricing and excludes caching, batch, "
        "network, platform, engineering, and operational costs."
    ]
    if source_cost.unknown_price_components or target_cost.unknown_price_components:
        caveats.append("A total or delta is unavailable because required pricing is unknown.")
    if (
        source_cost.total_cost_usd is not None
        and not source_cost.sources
        or target_cost.total_cost_usd is not None
        and not target_cost.sources
    ):
        caveats.append("A canonical price has no pricing-specific provenance reference.")
    return MigrationCostEstimate(
        workload=workload,
        source=source_cost,
        target=target_cost,
        estimated_delta_usd=delta,
        estimated_savings_percent=savings,
        caveats=caveats,
    )
