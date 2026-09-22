"""Deterministic model identifier resolution and lenient candidate matching."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from llm_migrate.core.models import (
    IdentifierMatchType,
    ModelCapabilities,
    ModelMatchCandidate,
    ModelMatchResult,
    ModelMatchStatus,
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
    matches.extend(
        (IdentifierMatchType.INVOCATION_SELECTOR, platform)
        for platform in profile.platforms
        if platform.invocation is not None
        for selector in platform.invocation.selectors
        if normalized == normalize_identifier(selector.model_id)
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
    platform_match_types = {
        IdentifierMatchType.PLATFORM_MODEL_ID,
        IdentifierMatchType.INVOCATION_SELECTOR,
    }
    # Deduped by identity: one representation matched through both its bare id
    # and a selector id must still count as exactly one implied platform.
    implied_platforms: list[PlatformAvailability] = []
    for item in profile_matches:
        if (
            item[1] in platform_match_types
            and item[2] is not None
            and all(existing is not item[2] for existing in implied_platforms)
        ):
            implied_platforms.append(item[2])
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
    selected_match = next(
        (
            item[1]
            for item in profile_matches
            if item[1] in platform_match_types and item[2] == selected
        ),
        None,
    )
    match_type = (
        selected_match if selected is not None and selected_match else profile_matches[0][1]
    )
    return ResolvedModel(
        profile=profile,
        matched_identifier=query,
        matched_by=match_type,
        platform=selected,
        effective_capabilities=effective_capabilities(profile, selected),
    )


# ---------------------------------------------------------------------------
# Lenient, registry-first matching for user-supplied identifiers
# ---------------------------------------------------------------------------

REGIONAL_PROFILE_PREFIXES = (
    "us.",
    "eu.",
    "apac.",
    "ap.",
    "jp.",
    "au.",
    "ca.",
    "sa.",
    "global.",
    "us-gov.",
)
_VERSION_SUFFIX = re.compile(r"[-.]v\d+(?::\d+)?$")

_FUZZY_THRESHOLD = 0.6
_MAX_CANDIDATES = 5

_CONFIRM_GUIDANCE = (
    "The identifier is not an exact registry match. Confirm the intended model "
    "with the user before proceeding, then retry with the confirmed canonical name."
)
_NOT_FOUND_GUIDANCE = (
    "No registry model resembles this identifier. Confirm the exact identifier with "
    "the user; if it is a genuinely new model, use the V1.1 research workflow "
    "(create_migration_research_request) instead of guessing."
)


def strip_identifier_decorations(value: str) -> str:
    """Remove platform routing decorations that do not change model identity.

    Bedrock cross-region inference-profile prefixes (`us.`, `eu.`, ...) and
    trailing version markers (`-v1:0`) route requests; they do not select a
    different underlying model.
    """
    text = value.strip().casefold()
    changed = True
    while changed:
        changed = False
        for prefix in REGIONAL_PROFILE_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix) :]
                changed = True
    return _VERSION_SUFFIX.sub("", text)


def normalize_platform(registry: ModelRegistry, value: str | None) -> tuple[str | None, str | None]:
    """Map a vague platform spelling onto one known registry platform.

    Returns (normalized platform or the raw value, explanatory note or None).
    """
    if value is None:
        return None, None
    known = sorted(
        {platform.platform for profile in registry.all() for platform in profile.platforms}
    )
    normalized_query = normalize_identifier(value)
    for platform in known:
        if normalize_identifier(platform) == normalized_query:
            return platform, None
    containing = [
        platform
        for platform in known
        if normalized_query in normalize_identifier(platform)
        or normalize_identifier(platform) in normalized_query
    ]
    if len(containing) == 1:
        return containing[0], f"Interpreted platform {value!r} as {containing[0]!r}."
    return value, (
        f"Platform {value!r} is not a known registry platform; known platforms: "
        + ", ".join(known)
        + "."
    )


def _identifier_pool(profile: ModelProfile) -> list[tuple[str, str | None]]:
    """Every identifier that may name this profile, with an optional platform."""
    identity = profile.identity
    pool: list[tuple[str, str | None]] = [
        (identity.canonical_name, None),
        (identity.display_name, None),
        *((alias, None) for alias in identity.aliases),
    ]
    pool.extend((platform.model_id, platform.platform) for platform in profile.platforms)
    pool.extend(
        (selector.model_id, platform.platform)
        for platform in profile.platforms
        if platform.invocation is not None
        for selector in platform.invocation.selectors
    )
    return pool


def _score_identifier(query: str, identifier: str) -> tuple[float, str]:
    """Deterministic similarity between one query and one registry identifier."""
    normalized_query = normalize_identifier(query)
    normalized_identifier = normalize_identifier(identifier)
    stripped_query = normalize_identifier(strip_identifier_decorations(query))
    stripped_identifier = normalize_identifier(strip_identifier_decorations(identifier))
    if normalized_query == normalized_identifier:
        return 1.0, "exact identifier match"
    if stripped_query and stripped_query == stripped_identifier:
        return 0.97, (
            "matches after removing regional inference-profile prefixes or version suffixes"
        )
    ratio = SequenceMatcher(None, normalized_query, normalized_identifier).ratio()
    if normalized_query and (
        normalized_query in normalized_identifier or normalized_identifier in normalized_query
    ):
        return max(ratio, 0.85), "one identifier contains the other"
    return ratio, "approximate spelling similarity"


def _best_candidate(
    profile: ModelProfile, query: str, platform: str | None
) -> ModelMatchCandidate | None:
    best: tuple[float, str, str, str | None] | None = None
    for identifier, identifier_platform in _identifier_pool(profile):
        score, reason = _score_identifier(query, identifier)
        if platform and identifier_platform and identifier_platform != platform:
            score *= 0.95
        key = (score, reason, identifier, identifier_platform)
        if best is None or key[0] > best[0]:
            best = key
    if best is None or best[0] < _FUZZY_THRESHOLD:
        return None
    score, reason, identifier, identifier_platform = best
    return ModelMatchCandidate(
        canonical_name=profile.identity.canonical_name,
        display_name=profile.identity.display_name,
        provider=profile.identity.provider,
        matched_identifier=identifier,
        platform=identifier_platform,
        similarity=round(score, 4),
        reason=reason,
    )


def _resolved_result(
    resolution: ResolvedModel,
    query: str,
    platform_query: str | None,
    notes: list[str],
) -> ModelMatchResult:
    return ModelMatchResult(
        query=query,
        platform_query=platform_query,
        status=ModelMatchStatus.RESOLVED,
        canonical_name=resolution.canonical_name,
        platform=resolution.platform.platform if resolution.platform else None,
        model_id=resolution.platform.model_id if resolution.platform else None,
        resolution=resolution,
        notes=notes,
        guidance="The identifier resolved against the local registry; no research is needed "
        "to identify this model.",
    )


def _representation_candidates(
    profile: ModelProfile, platform: str | None
) -> list[ModelMatchCandidate]:
    """One candidate per platform representation so the caller can pick exactly one."""
    identity = profile.identity
    representations = [
        item
        for item in profile.platforms
        if not platform or item.platform.casefold() == platform.casefold()
    ] or profile.platforms
    return [
        ModelMatchCandidate(
            canonical_name=identity.canonical_name,
            display_name=identity.display_name,
            provider=identity.provider,
            matched_identifier=item.model_id,
            platform=item.platform,
            endpoint=item.endpoint,
            similarity=1.0,
            reason="the identifier matches this model; select one platform/endpoint representation",
        )
        for item in representations
    ]


def matching_canonical_names(registry: ModelRegistry, query: str) -> set[str]:
    """Canonical names whose identifiers exactly match the query, however many."""
    normalized = normalize_identifier(query)
    return {
        profile.identity.canonical_name
        for profile in registry.all()
        if _matches(profile, normalized)
    }


def _sole_matching_profile(registry: ModelRegistry, query: str) -> ModelProfile | None:
    normalized = normalize_identifier(query)
    matching = [profile for profile in registry.all() if _matches(profile, normalized)]
    return matching[0] if len(matching) == 1 else None


def match_model(
    registry: ModelRegistry,
    query: str,
    platform: str | None = None,
    endpoint: str | None = None,
) -> ModelMatchResult:
    """Registry-first matching that suggests candidates instead of failing hard.

    Exact matches and deterministic decoration-equivalent matches resolve;
    anything else returns ranked candidates for explicit user confirmation.
    """
    notes: list[str] = []
    normalized_platform, platform_note = normalize_platform(registry, platform)
    if platform_note:
        notes.append(platform_note)

    def _attempt(identifier: str) -> ModelMatchResult | None:
        try:
            resolution = resolve_model(registry, identifier, normalized_platform, endpoint)
        except AmbiguousModelError as exc:
            notes.append(str(exc))
            profile = _sole_matching_profile(registry, identifier)
            if profile is None:
                return None
            return ModelMatchResult(
                query=query,
                platform_query=platform,
                status=ModelMatchStatus.NEEDS_CONFIRMATION,
                candidates=_representation_candidates(profile, normalized_platform),
                notes=notes,
                guidance=(
                    "The model is identified but has multiple platform/endpoint "
                    "representations. Confirm the platform and endpoint with the user, "
                    "then retry with both specified."
                ),
            )
        except ModelNotFoundError as exc:
            profile = _sole_matching_profile(registry, identifier)
            if profile is None:
                return None
            notes.append(str(exc))
            return ModelMatchResult(
                query=query,
                platform_query=platform,
                status=ModelMatchStatus.NEEDS_CONFIRMATION,
                candidates=_representation_candidates(profile, None),
                notes=notes,
                guidance=(
                    "The model is identified but the requested platform/endpoint context "
                    "does not match any of its representations. Confirm the platform "
                    "with the user, then retry."
                ),
            )
        if resolution.matched_by is IdentifierMatchType.INVOCATION_SELECTOR:
            notes.append(
                f"Identifier {query!r} is a reviewed invocation selector of "
                f"{resolution.canonical_name!r}: it routes on-demand invocation via an "
                "inference-profile id and does not change model identity."
            )
        elif identifier is not query:
            notes.append(
                f"Identifier {query!r} matched {resolution.canonical_name!r} after removing "
                "a regional inference-profile prefix or version suffix; these route "
                "requests and do not change model identity."
            )
        return _resolved_result(resolution, query, platform, notes)

    result = _attempt(query)
    if result is not None:
        return result
    stripped = strip_identifier_decorations(query)
    if normalize_identifier(stripped) != normalize_identifier(query):
        result = _attempt(stripped)
        if result is not None:
            return result
    candidates = sorted(
        (
            candidate
            for profile in registry.all()
            if (candidate := _best_candidate(profile, query, normalized_platform)) is not None
        ),
        key=lambda item: (-item.similarity, item.canonical_name),
    )[:_MAX_CANDIDATES]
    if candidates:
        return ModelMatchResult(
            query=query,
            platform_query=platform,
            status=ModelMatchStatus.NEEDS_CONFIRMATION,
            candidates=candidates,
            notes=notes,
            guidance=_CONFIRM_GUIDANCE,
        )
    return ModelMatchResult(
        query=query,
        platform_query=platform,
        status=ModelMatchStatus.NOT_FOUND,
        notes=notes,
        guidance=_NOT_FOUND_GUIDANCE,
    )
