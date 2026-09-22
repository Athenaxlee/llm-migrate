"""Configuration file scanning: prompt provenance and semantic config couplings.

Parses YAML/JSON/TOML documents found in an application without executing any
application code, finds configuration values that reference prompt files
(for example `prompts: {multi: prompt_lib/claude_prompt.yaml}`), and promotes
model-coupled configuration values (model ids, sampling, token budgets,
region/routing, pricing) to first-class couplings — but only when the document
is anchored: a registry-matched model id spelling appears in it, or its values
are traced into a detected invocation call chain.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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


# ---------------------------------------------------------------------------
# Semantic configuration couplings (V1.5.0-b)
# ---------------------------------------------------------------------------

ConfigValueKind = Literal["model_id", "sampling", "token_budget", "region", "pricing"]

_MODEL_ID_KEYS = {"model", "model_id", "modelid", "model_name"}
_SAMPLING_KEYS = {"temperature", "top_p", "top_k"}
_TOKEN_BUDGET_KEYS = {
    "max_tokens",
    "max_output_tokens",
    "max_input_tokens",
    "max_completion_tokens",
    "budget_tokens",
    "token_budget",
}
_REGION_KEYS = {"region", "region_name", "aws_region", "routing"}
_PRICING_KEY_MARKERS = ("price", "pricing", "cost")


@dataclass(frozen=True)
class ConfigCoupling:
    """One model-coupled configuration value in an anchored config document."""

    key_path: tuple[str, ...]
    value_kind: ConfigValueKind
    value: str | bool | int | float


def _value_kind(key: str, value: Any) -> ConfigValueKind | None:
    lowered = key.casefold()
    if (lowered in _MODEL_ID_KEYS or lowered.endswith(("_model", "_model_id"))) and isinstance(
        value, str
    ):
        return "model_id"
    if lowered in _SAMPLING_KEYS and isinstance(value, (int, float)):
        return "sampling"
    if lowered in _TOKEN_BUDGET_KEYS and isinstance(value, int):
        return "token_budget"
    if lowered in _REGION_KEYS and isinstance(value, str):
        return "region"
    if any(marker in lowered for marker in _PRICING_KEY_MARKERS) and isinstance(
        value, (int, float)
    ):
        return "pricing"
    return None


def find_config_couplings(
    document: ConfigDocument,
    known_model_ids: set[str],
    *,
    usage_anchored: bool,
) -> list[ConfigCoupling]:
    """Model-coupled values in one config document, with precision guardrails.

    A document contributes couplings only when the detection is anchored:
    (a) a registry-matched model id spelling appears among its string values,
    or (b) `usage_anchored` — the document's values were traced into a
    detected invocation call chain. Pricing-shaped values are promoted only
    under a model-id anchor (a): pricing alone never becomes a coupling.
    """
    candidates: list[ConfigCoupling] = []
    model_anchored = False

    def walk(node: Any, key_path: tuple[str, ...]) -> None:
        nonlocal model_anchored
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    walk(value, (*key_path, key))
        elif isinstance(node, list):
            for item in node:
                walk(item, key_path)
        elif isinstance(node, (str, bool, int, float)) and key_path:
            if isinstance(node, str) and node in known_model_ids:
                model_anchored = True
                candidates.append(
                    ConfigCoupling(key_path=key_path, value_kind="model_id", value=node)
                )
                return
            kind = _value_kind(key_path[-1], node)
            if kind is not None:
                candidates.append(ConfigCoupling(key_path=key_path, value_kind=kind, value=node))

    walk(document.data, ())
    if not model_anchored and not usage_anchored:
        return []
    return [
        coupling
        for coupling in candidates
        # Pricing-shaped values without a model anchor are never couplings.
        if coupling.value_kind != "pricing" or model_anchored
    ]
