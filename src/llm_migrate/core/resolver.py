"""Deterministic model identifier resolution."""

from __future__ import annotations

import re
import unicodedata

from llm_migrate.core.models import (
    IdentifierMatchType,
    ModelCapabilities,
    ModelProfile,
    PlatformAvailability,
    ResolvedModel,
)
from llm_migrate.core.registry import ModelRegistry, RegistryError


class ModelNotFoundError(RegistryError):
    """No registry identifier matches the input."""


class AmbiguousModelError(RegistryError):
    """More than one model claims the input identifier."""


def normalize_identifier(value: str) -> str:
    """Normalize punctuation and spacing without discarding meaningful digits."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())


def _matches(
    profile: ModelProfile, normalized: str
) -> list[tuple[IdentifierMatchType, PlatformAvailability | None]]:
    identity = profile.identity
    matches: list[tuple[IdentifierMatchType, PlatformAvailability | None]] = []
    if normalized == normalize_identifier(identity.canonical_name):
        matches.append((IdentifierMatchType.CANONICAL_NAME, None))
    if normalized == normalize_identifier(identity.display_name):
        matches.append((IdentifierMatchType.DISPLAY_NAME, None))
    if any(normalized == normalize_identifier(alias) for alias in identity.aliases):
        matches.append((IdentifierMatchType.ALIAS, None))
    matches.extend(
        (IdentifierMatchType.PLATFORM_MODEL_ID, platform)
        for platform in profile.platforms
        if normalized == normalize_identifier(platform.model_id)
    )
    return matches


def effective_capabilities(
    profile: ModelProfile, platform: PlatformAvailability | None
) -> ModelCapabilities:
    if platform is None or platform.capability_overrides is None:
        return profile.capabilities
    updates = platform.capability_overrides.model_dump(exclude_none=True)
    return profile.capabilities.model_copy(update=updates)


def resolve_model(
    registry: ModelRegistry,
    query: str,
    platform: str | None = None,
    endpoint: str | None = None,
) -> ResolvedModel:
    normalized = normalize_identifier(query)
    if not normalized:
        raise ModelNotFoundError("model identifier cannot be empty")
    profile_matches = [
        (profile, match_type, matched_platform)
        for profile in registry.all()
        for match_type, matched_platform in _matches(profile, normalized)
    ]
    matched_names = {item[0].identity.canonical_name for item in profile_matches}
    if not profile_matches:
        raise ModelNotFoundError(
            f"no model matches {query!r}; add an exact alias or platform model ID to the registry"
        )
    if len(matched_names) > 1:
        names = ", ".join(sorted(matched_names))
        raise AmbiguousModelError(f"ambiguous model identifier {query!r}; matches: {names}")
    profile = profile_matches[0][0]
    implied_platforms = [
        item[2]
        for item in profile_matches
        if item[1] is IdentifierMatchType.PLATFORM_MODEL_ID and item[2] is not None
    ]
    candidates = [
        item
        for item in profile.platforms
        if (not platform or item.platform.casefold() == platform.casefold())
        and (not endpoint or (item.endpoint or "").casefold() == endpoint.casefold())
    ]
    selected: PlatformAvailability | None = None
    if platform or endpoint:
        if not candidates:
            raise ModelNotFoundError(
                f"model {profile.identity.canonical_name!r} has no matching platform representation"
            )
        if len(candidates) > 1:
            choices = ", ".join(f"{item.platform}/{item.endpoint or '-'}" for item in candidates)
            raise AmbiguousModelError(
                f"platform representation is ambiguous for {query!r}; matches: {choices}"
            )
        selected = candidates[0]
    elif len(implied_platforms) == 1:
        selected = implied_platforms[0]
    elif len(implied_platforms) > 1:
        choices = ", ".join(
            sorted(f"{item.platform}/{item.endpoint or '-'}" for item in implied_platforms)
        )
        raise AmbiguousModelError(
            f"platform model ID {query!r} maps to multiple representations: {choices}; "
            "provide platform and endpoint"
        )
    if selected is not None and any(
        item[1] is IdentifierMatchType.PLATFORM_MODEL_ID and item[2] == selected
        for item in profile_matches
    ):
        match_type = IdentifierMatchType.PLATFORM_MODEL_ID
    else:
        match_type = profile_matches[0][1]
    return ResolvedModel(
        profile=profile,
        matched_identifier=query,
        matched_by=match_type,
        platform=selected,
        effective_capabilities=effective_capabilities(profile, selected),
    )
