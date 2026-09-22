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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import yaml

from llm_migrate.analyzers.prompt import structural_sections
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


def _is_model_id_swap(
    original: Any, adapted: Any, source_ids: Sequence[str], target_ids: Sequence[str]
) -> bool:
    """Whether a non-prompt value changed exactly by a source->target id swap.

    Any reviewed spelling of the source may be replaced by any reviewed
    spelling of the target (bare platform id or a selector-qualified
    invocation id); anything else remains an unsanctioned change.
    """
    original_json = json.dumps(original, sort_keys=True, default=str)
    adapted_json = json.dumps(adapted, sort_keys=True, default=str)
    return any(
        source_id in original_json and adapted_json == original_json.replace(source_id, target_id)
        for source_id in sorted(source_ids, key=len, reverse=True)
        for target_id in target_ids
        if source_id != target_id
    )


def _document_problems(
    original: Any,
    adapted: Any,
    original_components: list[PromptComponent],
    source_model_ids: Sequence[str],
    target_model_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Non-prompt values must be preserved exactly; only components may change.

    The one sanctioned non-prompt change is replacing the source model id with
    the target model id (a prompt document may carry its own `model:` value);
    such swaps are reported as notes, never as problems.
    """
    if not isinstance(original, dict):
        return [], []
    if not isinstance(adapted, dict):
        return (
            ["the adapted prompt document must keep the original top-level mapping structure"],
            [],
        )
    prompt_keys = {component.key for component in original_components if component.key}
    top_level_prompt_keys = {key.split("[", 1)[0] for key in prompt_keys}
    problems: list[str] = []
    notes: list[str] = []
    for key, value in original.items():
        if key in top_level_prompt_keys:
            continue
        if key not in adapted:
            problems.append(f"the adapted document dropped the non-prompt key {key!r}")
        elif adapted[key] != value:
            if (
                source_model_ids
                and target_model_ids
                and _is_model_id_swap(value, adapted[key], source_model_ids, target_model_ids)
            ):
                notes.append(
                    f"non-prompt value {key!r} updated from the source to the target model id"
                )
            else:
                problems.append(f"the adapted document changed the non-prompt value of {key!r}")
    for key in adapted:
        if key not in original:
            problems.append(f"the adapted document added the unexpected key {key!r}")
    return problems, notes


def _section_drops(label: str | None, original: str, adapted: str) -> list[str]:
    """Paired XML-like sections the adapted text lost relative to the original."""
    original_sections = structural_sections(original)
    adapted_sections = structural_sections(adapted)
    prefix = f"{label!r} " if label else ""
    drops: list[str] = []
    for name, count in original_sections.items():
        lost = count - adapted_sections.get(name, 0)
        if lost <= 0:
            continue
        detail = f"<{name}>" if count == 1 else f"{lost} of {count} <{name}> section(s)"
        drops.append(f"{prefix}drops structural section {detail}")
    return drops


def _component_drops(
    original_components: list[PromptComponent],
    adapted_components: list[PromptComponent],
) -> list[str]:
    """Removed prompt components and lost sections, tolerant of message reorder.

    Components with positional keys (`messages[i].content`) are compared in
    aggregate so that reordering, merging, or removing individual messages is
    not misreported as a structural loss when their sections survive.
    """
    named_adapted = {
        component.key: component
        for component in adapted_components
        if component.key and "[" not in component.key
    }
    drops: list[str] = []
    positional_original: list[PromptComponent] = []
    positional_adapted = [
        component for component in adapted_components if component.key and "[" in component.key
    ]
    for component in original_components:
        key = component.key
        if key is None:
            continue
        if "[" in key:
            positional_original.append(component)
            continue
        adapted_component = named_adapted.get(key)
        if adapted_component is None:
            drops.append(f"prompt component {key!r} was removed")
            continue
        drops.extend(_section_drops(key, component.content, adapted_component.content))
    if positional_original:
        original_joined = "\n".join(component.content for component in positional_original)
        adapted_joined = "\n".join(component.content for component in positional_adapted)
        if original_joined.strip() and not adapted_joined.strip():
            drops.append("all message content was removed or emptied")
        drops.extend(_section_drops("messages", original_joined, adapted_joined))
    return drops


@dataclass(frozen=True)
class SubmissionAssessment:
    """Deterministic checks over the DECODED runtime prompt values.

    Comparing decoded values (not serialized source text) means encoding
    tricks — unicode escapes, quoting changes, YAML style changes — can never
    make an unchanged prompt look adapted or hide content from validation.
    """

    problems: list[str]
    structural_drops: list[str]
    runtime_changed: bool
    cosmetic_only: bool
    checks_skipped: str | None = None
    notes: tuple[str, ...] = ()


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _runtime_views(
    components: list[PromptComponent],
) -> tuple[dict[str, str], list[str]]:
    """Named components as a mapping (order-insensitive) plus ordered messages.

    Top-level key order carries no runtime meaning (lookups are by key), so
    reordering keys is not a change; message order does carry meaning, so the
    positional components stay an ordered list.
    """
    named = {
        component.key: component.content
        for component in components
        if component.key and "[" not in component.key
    }
    positional = [
        component.content for component in components if component.key and "[" in component.key
    ]
    return named, positional


def evaluate_prompt_submission(
    original_text: str,
    adapted_text: str,
    format: PromptSourceFormat | None,
    *,
    source_model_id: str | None = None,
    target_model_id: str | None = None,
    source_model_ids: Sequence[str] | None = None,
    target_model_ids: Sequence[str] | None = None,
) -> SubmissionAssessment:
    """Fail-closed checks for a submitted prompt adaptation, in one parse.

    `problems` always block the submission (invalid syntax, changed non-prompt
    values); `structural_drops` block unless the submitter explicitly
    acknowledges a restructure; `runtime_changed` is False when the decoded
    prompt values are identical to the original (top-level key order ignored);
    `cosmetic_only` is True when they differ only by whitespace or letter
    case; `checks_skipped` discloses when the original could not be checked;
    `notes` records sanctioned model-id swaps in non-prompt values.
    """
    if format is None:
        changed = original_text != adapted_text
        return SubmissionAssessment(
            problems=[],
            structural_drops=_section_drops(None, original_text, adapted_text),
            runtime_changed=changed,
            cosmetic_only=changed and _normalized(original_text) == _normalized(adapted_text),
        )
    try:
        original = parse_structured_document(original_text, format)
    except PromptDocumentError as exc:
        # An unparseable original is not the submitter's problem, but silence
        # would make "could not check" look like "checked and passed".
        return SubmissionAssessment(
            [],
            [],
            True,
            False,
            checks_skipped=(
                "the original prompt document could not be parsed "
                f"({exc}); document and structural checks were skipped"
            ),
        )
    try:
        adapted = parse_structured_document(adapted_text, format)
    except PromptDocumentError as exc:
        return SubmissionAssessment(
            [f"the adapted prompt document is not valid {format.value}: {exc}"], [], True, False
        )
    original_components = extract_prompt_components(original)
    adapted_components = extract_prompt_components(adapted)
    original_named, original_positional = _runtime_views(original_components)
    adapted_named, adapted_positional = _runtime_views(adapted_components)
    changed = original_named != adapted_named or original_positional != adapted_positional
    cosmetic = (
        changed
        and {key: _normalized(value) for key, value in original_named.items()}
        == {key: _normalized(value) for key, value in adapted_named.items()}
        and [_normalized(value) for value in original_positional]
        == [_normalized(value) for value in adapted_positional]
    )
    source_ids = list(source_model_ids or ([source_model_id] if source_model_id else []))
    target_ids = list(target_model_ids or ([target_model_id] if target_model_id else []))
    problems, notes = _document_problems(
        original, adapted, original_components, source_ids, target_ids
    )
    return SubmissionAssessment(
        problems=problems,
        structural_drops=_component_drops(original_components, adapted_components),
        runtime_changed=changed,
        cosmetic_only=cosmetic,
        notes=tuple(notes),
    )


def document_mentions(text: str, format: PromptSourceFormat | None, needle: str) -> bool:
    """Whether the DECODED values of a prompt document mention `needle`.

    Structured documents are searched through their parsed string values, so
    escapes cannot hide a mention; unparseable or plain-text content falls
    back to a raw substring check.
    """
    if format in (PromptSourceFormat.YAML, PromptSourceFormat.JSON, PromptSourceFormat.TOML):
        try:
            data = parse_structured_document(text, format)
        except PromptDocumentError:
            return needle in text
        return needle in json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return needle in text


def is_prompt_bearing(text: str, format: PromptSourceFormat | None) -> bool:
    """Whether a file's content is prompt material that needs prompt validation."""
    if format in (PromptSourceFormat.YAML, PromptSourceFormat.JSON, PromptSourceFormat.TOML):
        try:
            return bool(extract_prompt_components(parse_structured_document(text, format)))
        except PromptDocumentError:
            return False
    return bool(structural_sections(text))
