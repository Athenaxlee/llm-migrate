"""Assemble prompt sources and discovery coverage from scanner evidence.

Confidence tiers:

- ``high``: scanned Python code provably loads the file (directly, or through
  a configuration value it reads), or the user explicitly named the file.
- ``medium``: a scanned configuration file references the file under a
  prompt-scoped key, but no scanned code was statically proven to load it;
  or a path resolved only by leading-segment stripping (never high); or a
  low candidate promoted by a unique key match with a dynamic consumer.
- ``low``: the file merely contains prompt-like keys; nothing references it.

Low-confidence candidates are reported for review but never become migration
tasks on their own. Promotions move one level at most (low -> medium), never
to high, and always record their exact evidence in the provenance chain.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from llm_migrate.core.models import (
    ApplicationFinding,
    CouplingKind,
    PromptDiscoveryCoverage,
    PromptDiscoverySummary,
    PromptSource,
    PromptSourceConfidence,
    PromptSourceOrigin,
)
from llm_migrate.core.prompt_documents import (
    STRUCTURED_SUFFIXES,
    extract_prompt_components,
    source_format,
    text_component,
)
from llm_migrate.scanners.config import (
    ConfigDocument,
    PathResolver,
    find_prompt_path_references,
    normalize_value,
    profile_model_id,
)

CONSUMER_DETAIL_PREFIX = "supplies prompt content"
OVERRIDE_PROVENANCE = "explicit prompt source override"
CONFIRMED_PROVENANCE = "user-confirmed prompt consumer {location} reads this file"


@dataclass(frozen=True)
class DiscoveredSource:
    """A code-discovered prompt source: its evidence chain and confidence cap."""

    provenance: list[str] = field(default_factory=list)
    confidence: PromptSourceConfidence = PromptSourceConfidence.HIGH


def is_prompt_consumer(finding: ApplicationFinding) -> bool:
    return finding.kind is CouplingKind.PROMPT and finding.detail.startswith(CONSUMER_DETAIL_PREFIX)


def consumer_location(finding: ApplicationFinding) -> str:
    """Stable `path:line:keyword` address of one prompt consumer.

    The keyword disambiguates several consumers on one line (`system=` and
    `messages=` of the same call), so a confirmation or dismissal never
    applies to a sibling consumer by accident.
    """
    return f"{finding.location.path}:{finding.location.line}:{finding.value}"


def resolve_override(raw: str, resolver: PathResolver) -> str | None:
    """Known-files spelling of a user-named application path, or None."""
    normalized = normalize_value(raw.strip())
    return resolver.canonical(normalized) if normalized is not None else None


def apply_consumer_decisions(
    findings: list[ApplicationFinding],
    confirmations: Mapping[str, str],
    dismissed: Collection[str],
) -> tuple[list[ApplicationFinding], list[str]]:
    """Fold recorded consumer confirmations and dismissals into the findings.

    A confirmed consumer becomes source-backed by the named file; a dismissed
    one (genuinely runtime-built content, by the user's recorded rationale)
    stops counting as dynamic. Addresses that match no current consumer are
    reported, never silently applied.
    """
    matched: set[str] = set()
    updated: list[ApplicationFinding] = []
    for finding in findings:
        location = consumer_location(finding) if is_prompt_consumer(finding) else None
        metadata = dict(finding.metadata or {})
        if location is not None and location in confirmations:
            matched.add(location)
            metadata.update(
                {"resolution": "source", "sources": [confirmations[location]], "confirmed": True}
            )
            finding = finding.model_copy(update={"metadata": metadata})
        elif location is not None and location in dismissed:
            matched.add(location)
            metadata.update({"resolution": "dismissed", "dismissed": True})
            finding = finding.model_copy(update={"metadata": metadata})
        updated.append(finding)
    warnings = [
        f"recorded prompt consumer decision for {location} matches no current prompt "
        "consumer; it was not applied"
        for location in sorted({*confirmations, *dismissed} - matched)
    ]
    return updated, warnings


def _build_source(
    base: Path,
    catalog: dict[str, ConfigDocument],
    path: str,
    confidence: PromptSourceConfidence,
    origin: PromptSourceOrigin,
    provenance: list[str],
    warnings: list[str],
) -> PromptSource | None:
    format = source_format(path)
    if format is None:
        warnings.append(f"prompt source {path!r} does not have a supported prompt file format")
        return None
    if Path(path).suffix.casefold() in STRUCTURED_SUFFIXES:
        document = catalog.get(path)
        if document is None:
            # Parse failures were already warned about during config scanning.
            return None
        components = extract_prompt_components(document.data)
    else:
        try:
            components = text_component((base / path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            warnings.append(f"prompt source {path!r} could not be read: {exc}")
            return None
    return PromptSource(
        path=path,
        format=format,
        components=components,
        provenance=provenance,
        confidence=confidence,
        origin=origin,
    )


def assemble_prompt_sources(
    base: Path,
    catalog: dict[str, ConfigDocument],
    resolver: PathResolver,
    discovered: dict[str, DiscoveredSource],
    overrides: Sequence[str] | None,
    confirmed: Mapping[str, Sequence[str]] | None = None,
) -> tuple[list[PromptSource], list[str]]:
    """Combine code-discovered, config-referenced, swept, and override sources.

    `confirmed` maps a source path to the consumer locations the user
    confirmed read it; such a source is included like an override, with the
    confirmation recorded as its provenance.
    """
    warnings: list[str] = []
    sources: dict[str, PromptSource] = {}
    for path in sorted(discovered):
        source = _build_source(
            base,
            catalog,
            path,
            discovered[path].confidence,
            PromptSourceOrigin.DISCOVERED,
            discovered[path].provenance,
            warnings,
        )
        if source is not None:
            sources[path] = source
    references = {
        config_path: find_prompt_path_references(catalog[config_path], resolver)
        for config_path in sorted(catalog)
    }
    references_by_target: dict[str, list[str | None]] = {}
    for config_path, found in references.items():
        for reference in found:
            if reference.prompt_scoped:
                references_by_target.setdefault(reference.target_path, []).append(
                    profile_model_id(catalog[config_path], reference.key_path)
                )
    for found in references.values():
        for reference in found:
            if reference.target_path in sources or not reference.prompt_scoped:
                continue
            rule = f"{reference.resolution.description}; " if reference.resolution else ""
            provenance = [
                f"{reference.config_path}: {'.'.join(reference.key_path)} -> "
                f"{reference.target_path} ({rule}config reference; loading code not "
                "statically proven)"
            ]
            source = _build_source(
                base,
                catalog,
                reference.target_path,
                PromptSourceConfidence.MEDIUM,
                PromptSourceOrigin.DISCOVERED,
                provenance,
                warnings,
            )
            if source is None:
                continue
            profile_ids = references_by_target.get(reference.target_path, [])
            if profile_ids and all(item is not None for item in profile_ids):
                source = source.model_copy(
                    update={"profile_model_ids": sorted({item for item in profile_ids if item})}
                )
            if not source.components:
                warnings.append(
                    f"{reference.config_path} references {reference.target_path} under "
                    f"{'.'.join(reference.key_path)}, but no prompt components were recognized "
                    "in it."
                )
                continue
            sources[reference.target_path] = source
    for config_path in sorted(catalog):
        if config_path in sources:
            continue
        components = extract_prompt_components(catalog[config_path].data)
        if not components:
            continue
        source = _build_source(
            base,
            catalog,
            config_path,
            PromptSourceConfidence.LOW,
            PromptSourceOrigin.DISCOVERED,
            ["prompt-like top-level keys; no reference from scanned code or configuration"],
            warnings,
        )
        if source is not None:
            sources[config_path] = source
    explicit: list[tuple[str, str]] = [(raw, OVERRIDE_PROVENANCE) for raw in overrides or []]
    for path, locations in sorted((confirmed or {}).items()):
        explicit.extend(
            (path, CONFIRMED_PROVENANCE.format(location=location)) for location in locations
        )
    for raw, reason in explicit:
        if normalize_value(raw.strip()) is None:
            warnings.append(f"prompt source override {raw!r} escapes the application root")
            continue
        resolved = resolve_override(raw, resolver)
        if resolved is None:
            warnings.append(f"prompt source override {raw!r} does not exist in the application")
            continue
        path = resolved
        existing = sources.get(path)
        if existing is not None:
            sources[path] = existing.model_copy(
                update={
                    "confidence": PromptSourceConfidence.HIGH,
                    "origin": PromptSourceOrigin.OVERRIDE,
                    "provenance": [reason, *existing.provenance],
                }
            )
            continue
        source = _build_source(
            base,
            catalog,
            path,
            PromptSourceConfidence.HIGH,
            PromptSourceOrigin.OVERRIDE,
            [reason],
            warnings,
        )
        if source is None:
            continue
        if not source.components:
            warnings.append(
                f"prompt source override {path!r} contains no recognizable prompt components; "
                "no prompt migration was prepared for it."
            )
        sources[path] = source
    return sorted(sources.values(), key=lambda item: item.path), warnings


def _top_level_keys(source: PromptSource, catalog: dict[str, ConfigDocument]) -> set[str]:
    document = catalog.get(source.path)
    if document is None or not isinstance(document.data, dict):
        return set()
    return {key for key in document.data if isinstance(key, str)}


def promote_key_matches(
    sources: list[PromptSource],
    findings: list[ApplicationFinding],
    catalog: dict[str, ConfigDocument],
) -> list[PromptSource]:
    """Promote a low candidate one level when it UNIQUELY matches a consumer.

    A dynamic consumer that reads literal keys (`prompts["sys_prompt"]`,
    `.get("user_prompt")`) matches every structured source whose top-level
    keys contain all of them. Exactly one match that is a low-confidence
    candidate is promoted to medium, with the consumer and keys recorded as
    the promotion evidence. Several matches are ambiguous (the application
    demonstrably selects among them at runtime) and promote nothing, and a
    consumer already matched by a medium/high source is served by it — the
    start-time confirmation and the discovery state handle those instead.
    """
    keyed = [(source, _top_level_keys(source, catalog)) for source in sources]
    promotions: dict[str, list[str]] = {}
    for finding in findings:
        if not is_prompt_consumer(finding):
            continue
        metadata = finding.metadata or {}
        access = metadata.get("access_keys") or []
        if metadata.get("resolution") != "dynamic" or not access:
            continue
        matches = [source for source, keys in keyed if keys and set(access) <= keys]
        if len(matches) != 1 or matches[0].confidence is not PromptSourceConfidence.LOW:
            continue
        promotions.setdefault(matches[0].path, []).append(
            f"key-match promotion: prompt consumer {consumer_location(finding)} reads "
            f"{', '.join(repr(key) for key in access)}, and {matches[0].path} is the only "
            "prompt document with those top-level keys"
        )
    promoted: list[PromptSource] = []
    for source in sources:
        evidence = promotions.get(source.path)
        if evidence is None or not source.components:
            promoted.append(source)
            continue
        promoted.append(
            source.model_copy(
                update={
                    "confidence": PromptSourceConfidence.MEDIUM,
                    "provenance": [*source.provenance, *evidence],
                }
            )
        )
    return promoted


def summarize_prompt_discovery(
    findings: list[ApplicationFinding], sources: list[PromptSource]
) -> PromptDiscoverySummary:
    """Deterministic coverage summary over consumers and resolved sources."""
    consumers = [item for item in findings if is_prompt_consumer(item)]

    def count(resolution: str) -> int:
        return sum(1 for item in consumers if (item.metadata or {}).get("resolution") == resolution)

    inline = count("inline")
    source_backed = count("source")
    dismissed = count("dismissed")
    dynamic = len(consumers) - inline - source_backed - dismissed
    resolved_sources = sum(
        source.confidence is not PromptSourceConfidence.LOW and bool(source.components)
        for source in sources
    )
    low_confidence = len(sources) - resolved_sources
    if not consumers or dynamic == 0:
        coverage = PromptDiscoveryCoverage.RESOLVED
    elif inline or source_backed or dismissed or resolved_sources:
        coverage = PromptDiscoveryCoverage.PARTIAL
    else:
        coverage = PromptDiscoveryCoverage.UNRESOLVED
    return PromptDiscoverySummary(
        consumers=len(consumers),
        inline_consumers=inline,
        source_backed_consumers=source_backed,
        dynamic_consumers=dynamic,
        dismissed_consumers=dismissed,
        resolved_sources=resolved_sources,
        low_confidence_sources=low_confidence,
        coverage=coverage,
    )
