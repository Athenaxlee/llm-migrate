"""Process-local, explicit evaluator allowlist for trusted host applications."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import EntryPoint, entry_points

from llm_migrate.core.evaluation import CustomEvaluator

_REGISTERED: dict[str, CustomEvaluator] = {}
_ENTRY_POINT_GROUP = "llm_migrate.evaluators"


def register_evaluator(name: str, evaluator: CustomEvaluator) -> None:
    """Register trusted code by name; corpus and MCP inputs can only reference the name."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
        raise ValueError("evaluator names must contain only letters, numbers, '.', '-', or '_'")
    if name in _REGISTERED and _REGISTERED[name] is not evaluator:
        raise ValueError(f"evaluator {name!r} is already registered")
    _REGISTERED[name] = evaluator


def unregister_evaluator(name: str) -> None:
    """Remove a process-local evaluator registration."""
    _REGISTERED.pop(name, None)


def registered_evaluators() -> Mapping[str, CustomEvaluator]:
    """Return a defensive copy of the trusted evaluator allowlist."""
    return dict(_REGISTERED)


def configure_evaluators(
    names: Iterable[str],
    *,
    entry_point_provider: Callable[[], Iterable[EntryPoint]] | None = None,
) -> None:
    """Load only explicitly allowlisted installed evaluator entry points."""
    requested = list(dict.fromkeys(name.strip() for name in names if name.strip()))
    if not requested:
        return
    available = {
        point.name: point
        for point in (
            entry_point_provider()
            if entry_point_provider is not None
            else entry_points(group=_ENTRY_POINT_GROUP)
        )
    }
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("configured evaluator entry points were not found: " + ", ".join(unknown))
    for name in requested:
        try:
            evaluator = available[name].load()
        except Exception as exc:
            raise ValueError(
                f"configured evaluator {name!r} failed to load with {type(exc).__name__}"
            ) from exc
        if not callable(evaluator):
            raise ValueError(f"configured evaluator {name!r} is not callable")
        register_evaluator(name, evaluator)
