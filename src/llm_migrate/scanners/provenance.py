"""Assemble prompt sources and discovery coverage from scanner evidence.

Confidence tiers:

- ``high``: scanned Python code provably loads the file (directly, or through
  a configuration value it reads), or the user explicitly named the file.
- ``medium``: a scanned configuration file references the file under a
  prompt-scoped key, but no scanned code was statically proven to load it.
- ``low``: the file merely contains prompt-like keys; nothing references it.

Low-confidence candidates are reported for review but never become migration
tasks on their own.
"""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
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
from llm_migrate.scanners.config import ConfigDocument, find_prompt_path_references

CONSUMER_DETAIL_PREFIX = "supplies prompt content"
OVERRIDE_PROVENANCE = "explicit prompt source override"


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
    known_files: set[str],
    discovered: dict[str, list[str]],
    overrides: Sequence[str] | None,
) -> tuple[list[PromptSource], list[str]]:
    """Combine code-discovered, config-referenced, swept, and override sources."""
    warnings: list[str] = []
    sources: dict[str, PromptSource] = {}
    for path in sorted(discovered):
        source = _build_source(
            base,
            catalog,
            path,
            PromptSourceConfidence.HIGH,
            PromptSourceOrigin.DISCOVERED,
            discovered[path],
            warnings,
        )
        if source is not None:
            sources[path] = source
    for config_path in sorted(catalog):
        for reference in find_prompt_path_references(catalog[config_path], known_files):
            if reference.target_path in sources or not reference.prompt_scoped:
                continue
            provenance = [
                f"{reference.config_path}: {'.'.join(reference.key_path)} -> "
                f"{reference.target_path} (config reference; loading code not statically proven)"
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
    for raw in overrides or []:
        path = posixpath.normpath(raw.strip().replace("\\", "/"))
        if path.startswith(("..", "/")) or not path or path == ".":
            warnings.append(f"prompt source override {raw!r} escapes the application root")
            continue
        if path not in known_files:
            warnings.append(f"prompt source override {raw!r} does not exist in the application")
            continue
        existing = sources.get(path)
        if existing is not None:
            sources[path] = existing.model_copy(
                update={
                    "confidence": PromptSourceConfidence.HIGH,
                    "origin": PromptSourceOrigin.OVERRIDE,
                    "provenance": [OVERRIDE_PROVENANCE, *existing.provenance],
                }
            )
            continue
        source = _build_source(
            base,
            catalog,
            path,
            PromptSourceConfidence.HIGH,
            PromptSourceOrigin.OVERRIDE,
            [OVERRIDE_PROVENANCE],
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


def summarize_prompt_discovery(
    findings: list[ApplicationFinding], sources: list[PromptSource]
) -> PromptDiscoverySummary:
    """Deterministic coverage summary over consumers and resolved sources."""
    consumers = [
        item
        for item in findings
        if item.kind is CouplingKind.PROMPT and item.detail.startswith(CONSUMER_DETAIL_PREFIX)
    ]

    def count(resolution: str) -> int:
        return sum(1 for item in consumers if (item.metadata or {}).get("resolution") == resolution)

    inline = count("inline")
    source_backed = count("source")
    dynamic = len(consumers) - inline - source_backed
    resolved_sources = sum(
        source.confidence is not PromptSourceConfidence.LOW and bool(source.components)
        for source in sources
    )
    low_confidence = len(sources) - resolved_sources
    if not consumers or dynamic == 0:
        coverage = PromptDiscoveryCoverage.RESOLVED
    elif inline or source_backed or resolved_sources:
        coverage = PromptDiscoveryCoverage.PARTIAL
    else:
        coverage = PromptDiscoveryCoverage.UNRESOLVED
    return PromptDiscoverySummary(
        consumers=len(consumers),
        inline_consumers=inline,
        source_backed_consumers=source_backed,
        dynamic_consumers=dynamic,
        resolved_sources=resolved_sources,
        low_confidence_sources=low_confidence,
        coverage=coverage,
    )
