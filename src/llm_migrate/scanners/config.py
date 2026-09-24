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
from dataclasses import dataclass, field
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
class Resolution:
    """A path value resolved to one known application file, and how."""

    path: str
    rule: Literal["config_dir", "application_root", "code_dir", "stripped_prefix"]
    stripped: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        """Human-readable resolution rule for provenance chains."""
        if self.rule == "stripped_prefix":
            return (
                f"resolved by stripping leading {'/'.join(self.stripped)!r} "
                "(repo-root-relative value; matches the application root's trailing path)"
            )
        return {
            "config_dir": "resolved relative to the configuration file",
            "application_root": "resolved relative to the application root",
            "code_dir": "resolved relative to the loading module",
        }[self.rule]


_MAX_STRIPPED_SEGMENTS = 3


@dataclass(frozen=True)
class PathResolver:
    """Resolve path values to files INSIDE the application, never outside it.

    Resolution is always a lookup in `known_files` (the scanned file set), so
    no rule can reach a file outside the application. Values are normalized
    first (backslashes become `/`). On a case-insensitive filesystem —
    detected once per scan by `detect_case_insensitive`, never assumed from
    the OS — lookups and the stripping rule's segment comparison are
    case-folded, and the resolved path is always the `known_files` spelling.

    Bases are tried in order (the config file's directory, then the
    application root), then bounded leading-segment stripping: for k = 1..3,
    drop the value's first k segments when they equal the last k components
    of the application root's real path (a value written relative to a repo
    root above the scanned directory) and accept the remainder when known.
    """

    known_files: frozenset[str]
    root_tail: tuple[str, ...] = ()
    case_insensitive: bool = False
    _folded: dict[str, str | None] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        for item in self.known_files:
            key = item.casefold()
            # Two spellings folding together cannot both exist on a
            # case-insensitive filesystem; stay unresolved if they do.
            self._folded[key] = None if key in self._folded else item

    @classmethod
    def for_files(
        cls,
        known_files: set[str] | frozenset[str],
        root: Path | None = None,
        *,
        case_insensitive: bool = False,
    ) -> PathResolver:
        # parts[0] of an absolute path is its anchor ("/" or "C:\\").
        tail = tuple(root.resolve().parts[1:][-_MAX_STRIPPED_SEGMENTS:]) if root else ()
        return cls(frozenset(known_files), tail, case_insensitive)

    def canonical(self, path: str) -> str | None:
        """The known-files spelling of one normalized relative path, if known."""
        if path in self.known_files:
            return path
        if self.case_insensitive:
            return self._folded.get(path.casefold())
        return None

    def _same(self, left: str, right: str) -> bool:
        return left.casefold() == right.casefold() if self.case_insensitive else left == right

    def resolve(self, value: str, bases: tuple[tuple[str, str], ...]) -> Resolution | None:
        """Resolve one raw value against (rule, directory) bases, then stripping.

        A value may climb with `..` relative to a base directory (a config in
        `config/` naming `../prompts/x.yaml`); escaping is judged on the
        JOINED path, and the result is still a known-files lookup. Stripping
        applies only to values without `..`.
        """
        loose = normalize_value(value, allow_parent=True)
        if loose is None or source_format(loose) is None:
            return None
        for rule, prefix in bases:
            joined = posixpath.normpath(posixpath.join(prefix, loose)) if prefix else loose
            if joined.startswith(".."):
                continue
            found = self.canonical(joined)
            if found is not None:
                return Resolution(path=found, rule=rule)  # type: ignore[arg-type]
        normalized = normalize_value(value)
        if normalized is None:
            return None
        segments = normalized.split("/")
        for count in range(1, min(_MAX_STRIPPED_SEGMENTS, len(segments) - 1) + 1):
            if count > len(self.root_tail):
                break
            dropped = segments[:count]
            expected = self.root_tail[-count:]
            if not all(self._same(a, b) for a, b in zip(dropped, expected, strict=True)):
                continue
            found = self.canonical("/".join(segments[count:]))
            if found is not None:
                return Resolution(path=found, rule="stripped_prefix", stripped=tuple(dropped))
        return None


def normalize_value(value: str, *, allow_parent: bool = False) -> str | None:
    """Normalize a config/code path value; None for absolute or unusable values.

    `allow_parent` keeps a leading `..` (for joining with a base directory);
    the default rejects it, which is what every root-relative lookup needs.
    """
    if not value or value != value.strip() or "\n" in value:
        return None
    candidate = value.replace("\\", "/")
    if candidate.startswith(("/", "~")) or (len(candidate) > 1 and candidate[1] == ":"):
        return None
    normalized = posixpath.normpath(candidate)
    if normalized == "." or (normalized.startswith("..") and not allow_parent):
        return None
    return normalized


def detect_case_insensitive(base: Path, relative_files: list[str]) -> bool:
    """Whether the filesystem holding `base` folds case, probed on real files.

    Checks one scanned file whose name has letters: if its case-swapped
    spelling names the same file, the filesystem is case-insensitive. Never
    writes anything and never assumes from the operating system.
    """
    for relative in relative_files:
        name = posixpath.basename(relative)
        swapped = name.swapcase()
        if swapped == name:
            continue
        original = base / relative
        probe = original.with_name(swapped)
        try:
            return probe.exists() and probe.samefile(original)
        except OSError:
            return False
    return False


@dataclass(frozen=True)
class ConfigPathReference:
    """A config value that names an existing candidate prompt file."""

    config_path: str
    key_path: tuple[str, ...]
    target_path: str
    resolution: Resolution | None = None

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


def resolve_reference(
    config_path: str, value: str, resolver: PathResolver | set[str] | frozenset[str]
) -> Resolution | None:
    """Resolve a config string value to a known prompt-bearing file, if any.

    Values are tried relative to the config file's directory first, then
    relative to the application base, then by bounded leading-segment
    stripping (see `PathResolver`). Absolute paths and path traversal
    outside the application are never resolved.
    """
    if not isinstance(resolver, PathResolver):
        resolver = PathResolver(frozenset(resolver))
    return resolver.resolve(
        value,
        (("config_dir", posixpath.dirname(config_path)), ("application_root", "")),
    )


def find_prompt_path_references(
    document: ConfigDocument, resolver: PathResolver | set[str] | frozenset[str]
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
            resolution = resolve_reference(document.path, node, resolver)
            if resolution is not None:
                references.append(
                    ConfigPathReference(
                        config_path=document.path,
                        key_path=key_path,
                        target_path=resolution.path,
                        resolution=resolution,
                    )
                )

    walk(document.data, ())
    return references


def profile_model_id(document: ConfigDocument, key_path: tuple[str, ...]) -> str | None:
    """The model id declared by the nearest enclosing profile of a config value.

    Walks from the value's parent mapping toward the document root and
    returns the first sibling value under a model-id key (`model`,
    `model_id`, `*_model`, ...). None when no enclosing mapping declares one
    — the reference then cannot be scoped to a model.
    """
    for depth in range(len(key_path) - 1, -1, -1):
        node = lookup(document, key_path[:depth])
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if isinstance(key, str) and _value_kind(key, value) == "model_id":
                return str(value)
    return None


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
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if lowered in _SAMPLING_KEYS and numeric:
        return "sampling"
    if lowered in _TOKEN_BUDGET_KEYS and numeric and isinstance(value, int):
        return "token_budget"
    if lowered in _REGION_KEYS and isinstance(value, str):
        return "region"
    if any(marker in lowered for marker in _PRICING_KEY_MARKERS) and numeric:
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
