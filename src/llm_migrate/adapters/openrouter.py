"""Opt-in OpenRouter pricing evidence adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.request import Request, urlopen

from llm_migrate.core.models import LivePricingResult, ModelProfile, PricingMatchStatus

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Consequential matches are explicit. Missing entries return ``unmapped``; IDs are never guessed.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
}
ModelsFetcher = Callable[[str, float], bytes]


def _fetch(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "llm-migrate"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoint
        return bytes(response.read())


def _registry_prices(profile: ModelProfile) -> tuple[Decimal | None, Decimal | None]:
    pricing = profile.pricing
    return (
        pricing.input.amount if pricing and pricing.input else None,
        pricing.output.amount if pricing and pricing.output else None,
    )


def query_live_pricing(
    profile: ModelProfile,
    *,
    timeout: float = 5.0,
    model_ids: dict[str, str] | None = None,
    fetcher: ModelsFetcher | None = None,
    retrieved_at: datetime | None = None,
) -> LivePricingResult:
    """Return time-stamped OpenRouter-scoped evidence without mutating registry facts."""
    timestamp = retrieved_at or datetime.now(UTC)
    mapping = DEFAULT_MODEL_IDS if model_ids is None else model_ids
    openrouter_id = mapping.get(profile.identity.canonical_name)
    registry_input, registry_output = _registry_prices(profile)
    base: dict[str, Any] = {
        "canonical_name": profile.identity.canonical_name,
        "openrouter_model_id": openrouter_id,
        "retrieved_at": timestamp,
        "registry_input_per_million": registry_input,
        "registry_output_per_million": registry_output,
    }
    if openrouter_id is None:
        return LivePricingResult(status=PricingMatchStatus.UNMAPPED, **base)
    if timeout < 0.1 or timeout > 30.0:
        return LivePricingResult(
            status=PricingMatchStatus.UNAVAILABLE,
            error="OpenRouter pricing unavailable: timeout must be between 0.1 and 30 seconds",
            **base,
        )
    try:
        payload = json.loads((fetcher or _fetch)(OPENROUTER_MODELS_URL, timeout))
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("response data is not an array")
        matches = [
            item for item in data if isinstance(item, dict) and item.get("id") == openrouter_id
        ]
        if not matches:
            return LivePricingResult(status=PricingMatchStatus.NOT_FOUND, **base)
        if len(matches) != 1:
            raise ValueError("response contains duplicate exact model IDs")
        pricing = matches[0].get("pricing")
        if not isinstance(pricing, dict):
            raise ValueError("matching model has no pricing object")
        raw_prompt = pricing.get("prompt")
        raw_completion = pricing.get("completion")
        if not isinstance(raw_prompt, str) or not isinstance(raw_completion, str):
            raise ValueError("pricing.prompt and pricing.completion must be decimal strings")
        input_price = Decimal(raw_prompt) * Decimal(1_000_000)
        output_price = Decimal(raw_completion) * Decimal(1_000_000)
        discrepancies: list[str] = []
        if registry_input is not None and input_price != registry_input:
            discrepancies.append(
                f"OpenRouter input {input_price} differs from registry input {registry_input}"
            )
        if registry_output is not None and output_price != registry_output:
            discrepancies.append(
                f"OpenRouter output {output_price} differs from registry output {registry_output}"
            )
        return LivePricingResult(
            status=PricingMatchStatus.MATCHED,
            raw_prompt_per_token=raw_prompt,
            raw_completion_per_token=raw_completion,
            input_per_million=input_price,
            output_per_million=output_price,
            discrepancies=discrepancies,
            **base,
        )
    except (OSError, TimeoutError, ValueError, InvalidOperation) as exc:
        return LivePricingResult(
            status=PricingMatchStatus.UNAVAILABLE,
            error=f"OpenRouter pricing unavailable: {exc}",
            **base,
        )


def query_live_pricing_many(
    profiles: list[ModelProfile],
    *,
    timeout: float = 5.0,
    model_ids: dict[str, str] | None = None,
    fetcher: ModelsFetcher | None = None,
) -> list[LivePricingResult]:
    """Query one catalog snapshot and apply it to multiple exact model mappings."""
    timestamp = datetime.now(UTC)
    attempted = False
    cached: bytes | None = None
    failure: Exception | None = None
    underlying = fetcher or _fetch

    def cached_fetch(url: str, request_timeout: float) -> bytes:
        nonlocal attempted, cached, failure
        if not attempted:
            attempted = True
            try:
                cached = underlying(url, request_timeout)
            except Exception as exc:  # normalized by query_live_pricing
                failure = exc
        if failure is not None:
            raise OSError(str(failure)) from failure
        if cached is None:
            raise OSError("OpenRouter returned no response data")
        return cached

    return [
        query_live_pricing(
            profile,
            timeout=timeout,
            model_ids=model_ids,
            fetcher=cached_fetch,
            retrieved_at=timestamp,
        )
        for profile in profiles
    ]
