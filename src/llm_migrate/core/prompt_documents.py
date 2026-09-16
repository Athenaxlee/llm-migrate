"""Structured prompt document semantics: parsing, components, reconstruction.

A prompt source document is a file that carries prompt content, either as plain
text (`.txt`, `.md`) or as a structured document (`.yaml`, `.json`, `.toml`)
whose top-level keys name individual prompt components. This module owns the
deterministic rules for recognizing components, rebuilding a structured
document with migrated component content, and checking that a submitted
adaptation preserved every non-prompt value.
"""

from __future__ import annotations

import json
import tomllib
from typing import Any, Literal

import yaml

from llm_migrate.core.models import PromptComponent, PromptSourceFormat

STRUCTURED_SUFFIXES: dict[str, PromptSourceFormat] = {
    ".yaml": PromptSourceFormat.YAML,
    ".yml": PromptSourceFormat.YAML,
    ".json": PromptSourceFormat.JSON,
    ".toml": PromptSourceFormat.TOML,
}
TEXT_SUFFIXES: dict[str, PromptSourceFormat] = {
    ".txt": PromptSourceFormat.TEXT,
    ".md": PromptSourceFormat.MARKDOWN,
}
PROMPT_SOURCE_SUFFIXES = {**STRUCTURED_SUFFIXES, **TEXT_SUFFIXES}

_ROLES = Literal["system", "user", "developer", "unknown"]
PROMPT_KEY_ROLES: dict[str, _ROLES] = {
    "system": "system",
    "system_prompt": "system",
    "sys_prompt": "system",
    "developer": "developer",
    "developer_prompt": "developer",
    "user": "user",
    "user_prompt": "user",
    "prompt": "unknown",
}
_MESSAGE_ROLE_MAP: dict[str, _ROLES] = {
    "system": "system",
    "developer": "developer",
    "user": "user",
}


class PromptDocumentError(ValueError):
    """A prompt source document cannot be parsed or rebuilt as requested."""


def source_format(path: str) -> PromptSourceFormat | None:
    """Format for a prompt-bearing file path, or None when not prompt-bearing."""
    suffix = "." + path.rsplit(".", 1)[-1].casefold() if "." in path else ""
    return PROMPT_SOURCE_SUFFIXES.get(suffix)


def parse_structured_document(text: str, format: PromptSourceFormat) -> Any:
    """Parse one structured document; raises PromptDocumentError on bad syntax."""
    try:
        if format is PromptSourceFormat.YAML:
            return yaml.safe_load(text)
        if format is PromptSourceFormat.JSON:
            return json.loads(text)
        if format is PromptSourceFormat.TOML:
            return tomllib.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PromptDocumentError(f"invalid {format.value} document: {exc}") from exc
    raise PromptDocumentError(f"{format.value} is not a structured prompt document format")


def extract_prompt_components(data: Any) -> list[PromptComponent]:
    """Recognize prompt components among the top-level keys of a parsed document.

    Only well-known prompt key spellings count; arbitrary keys never become
    prompt components, so unrelated configuration is preserved untouched.
    """
    if not isinstance(data, dict):
        return []
    components: list[PromptComponent] = []
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        if key.casefold() == "messages":
            if isinstance(value, list):
                components.extend(_message_components(key, value))
            continue
        role = PROMPT_KEY_ROLES.get(key.casefold())
        if role is not None and isinstance(value, str) and value.strip():
            components.append(PromptComponent(role=role, key=key, content=value))
    return components


def _message_components(key: str, value: list[Any]) -> list[PromptComponent]:
    components: list[PromptComponent] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not (isinstance(content, str) and content.strip()):
            continue
        role = item.get("role")
        components.append(
            PromptComponent(
                role=_MESSAGE_ROLE_MAP.get(role, "unknown") if isinstance(role, str) else "unknown",
                key=f"{key}[{index}].content",
                content=content,
            )
        )
    return components


def text_component(text: str) -> list[PromptComponent]:
    """Whole-file component for plain-text prompt sources."""
    return [PromptComponent(role="unknown", key=None, content=text)] if text.strip() else []


class _LiteralDumper(yaml.SafeDumper):
    """Keep multiline prompt strings as literal blocks when rebuilding YAML."""


def _represent_multiline_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    if "\n" in value:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


_LiteralDumper.add_representer(str, _represent_multiline_str)


def build_candidate_document(
    original_text: str,
    format: PromptSourceFormat,
    replacements: dict[str, str],
) -> str | None:
    """Rebuild a structured document with component content replaced by key.

    Non-prompt values are carried over unchanged. Returns None when the format
    cannot be deterministically rebuilt (TOML has no stdlib writer).
    """
    data = parse_structured_document(original_text, format)
    if not isinstance(data, dict):
        return None
    updated = dict(data)
    for key, content in replacements.items():
        if key in updated and isinstance(updated[key], str):
            updated[key] = content
    if format is PromptSourceFormat.YAML:
        return yaml.dump(updated, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True)
    if format is PromptSourceFormat.JSON:
        return json.dumps(updated, indent=2, ensure_ascii=False) + "\n"
    return None


def structured_submission_problems(
    original_text: str,
    adapted_text: str,
    format: PromptSourceFormat,
) -> list[str]:
    """Fail-closed checks for a submitted structured prompt adaptation.

    The adapted document must parse, and every non-prompt value must be
    preserved exactly; only recognized prompt components may change.
    """
    try:
        original = parse_structured_document(original_text, format)
    except PromptDocumentError:
        return []  # An unparseable original is not the submitter's problem.
    try:
        adapted = parse_structured_document(adapted_text, format)
    except PromptDocumentError as exc:
        return [f"the adapted prompt document is not valid {format.value}: {exc}"]
    if not isinstance(original, dict):
        return []
    if not isinstance(adapted, dict):
        return ["the adapted prompt document must keep the original top-level mapping structure"]
    prompt_keys = {
        component.key for component in extract_prompt_components(original) if component.key
    }
    top_level_prompt_keys = {key.split("[", 1)[0] for key in prompt_keys}
    problems: list[str] = []
    for key, value in original.items():
        if key in top_level_prompt_keys:
            continue
        if key not in adapted:
            problems.append(f"the adapted document dropped the non-prompt key {key!r}")
        elif adapted[key] != value:
            problems.append(f"the adapted document changed the non-prompt value of {key!r}")
    for key in adapted:
        if key not in original:
            problems.append(f"the adapted document added the unexpected key {key!r}")
    return problems
