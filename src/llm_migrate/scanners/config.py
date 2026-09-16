"""Configuration file scanning for prompt provenance discovery.

Parses YAML/JSON/TOML documents found in an application without executing any
application code, and finds configuration values that reference prompt files
(for example `prompts: {multi: prompt_lib/claude_prompt.yaml}`).
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_migrate.core.models import PromptSourceFormat
from llm_migrate.core.prompt_documents import (
    STRUCTURED_SUFFIXES,
    PromptDocumentError,
    parse_structured_document,
    source_format,
)


@dataclass(frozen=True)
class ConfigDocument:
    """One parsed structured configuration document."""

    path: str
    format: PromptSourceFormat
    data: Any


@dataclass(frozen=True)
class ConfigPathReference:
    """A config value that names an existing candidate prompt file."""

    config_path: str
    key_path: tuple[str, ...]
    target_path: str

    @property
    def prompt_scoped(self) -> bool:
        """Whether any key on the path to the value mentions prompts."""
        return any("prompt" in key.casefold() for key in self.key_path)


def load_config_documents(
    base: Path, relative_paths: list[str]
) -> tuple[dict[str, ConfigDocument], list[str]]:
    """Parse every structured config file; unparseable files become warnings."""
    catalog: dict[str, ConfigDocument] = {}
    warnings: list[str] = []
    for relative in relative_paths:
        suffix = Path(relative).suffix.casefold()
        format = STRUCTURED_SUFFIXES.get(suffix)
        if format is None:
            continue
        try:
            text = (base / relative).read_text(encoding="utf-8")
            data = parse_structured_document(text, format)
        except (OSError, UnicodeError, PromptDocumentError) as exc:
            warnings.append(f"{relative}: {exc}")
            continue
        catalog[relative] = ConfigDocument(path=relative, format=format, data=data)
    return catalog, warnings


def resolve_reference(config_path: str, value: str, known_files: set[str]) -> str | None:
    """Resolve a config string value to a known prompt-bearing file, if any.

    Values are tried relative to the config file's directory first, then
    relative to the application base. Absolute paths and path traversal
    outside the application are never resolved.
    """
    if not value or value != value.strip() or "\n" in value or value.startswith(("/", "~")):
        return None
    if source_format(value) is None:
        return None
    for prefix in (posixpath.dirname(config_path), ""):
        candidate = posixpath.normpath(posixpath.join(prefix, value)) if prefix else value
        if candidate.startswith(".."):
            continue
        if candidate in known_files:
            return candidate
    return None


def find_prompt_path_references(
    document: ConfigDocument, known_files: set[str]
) -> list[ConfigPathReference]:
    """Find config values under prompt-scoped keys that name existing files."""
    references: list[ConfigPathReference] = []

    def walk(node: Any, key_path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    walk(value, (*key_path, key))
        elif isinstance(node, list):
            for item in node:
                walk(item, key_path)
        elif isinstance(node, str) and any("prompt" in key.casefold() for key in key_path):
            target = resolve_reference(document.path, node, known_files)
            if target is not None:
                references.append(
                    ConfigPathReference(
                        config_path=document.path,
                        key_path=key_path,
                        target_path=target,
                    )
                )

    walk(document.data, ())
    return references


def lookup(document: ConfigDocument, key_path: tuple[str, ...]) -> Any:
    """Follow one literal key path through a parsed document; None when absent."""
    node = document.data
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node
