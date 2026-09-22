"""Invocation identity: which id the platform actually accepts on demand.

Canonical identity (which model this is) and the invocation selector (which
id the platform accepts for an on-demand call) are different facts. Matching
correctly normalizes regional inference-profile prefixes away, but deployment
must reference an invocable id: a platform representation whose reviewed
`invocation` facts state `bare_on_demand_supported: false` cannot be invoked
by its bare `model_id` at all. This module derives the invocation choice for
one platform representation (seeded by the user's original spelling), builds
the spelling set every identity check matches against, and detects bare-id
references that the reviewed profile forbids.
"""

from __future__ import annotations

from pydantic import Field

from llm_migrate.core.models import (
    InvocationSelector,
    PlatformAvailability,
    StrictModel,
)
from llm_migrate.core.resolver import normalize_identifier


class InvocationChoice(StrictModel):
    """The invocation id one run side will reference, and how it was chosen."""

    invocation_model_id: str
    selector: str | None = None
    requires_selector: bool = False
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class InvocationDerivation(StrictModel):
    """Either a derived choice, or the selectors the user must pick from."""

    choice: InvocationChoice | None = None
    selection_required: list[InvocationSelector] = Field(default_factory=list)


def selector_for_spelling(
    platform: PlatformAvailability, spelling: str | None
) -> InvocationSelector | None:
    """The selector the user's original spelling names, if any.

    A spelling selects a selector when it equals the selector's full model id
    (up to punctuation/spacing normalization) or starts with the selector's
    `<name>.` prefix.
    """
    if not spelling or platform.invocation is None:
        return None
    normalized = normalize_identifier(spelling)
    text = spelling.strip().casefold()
    for selector in platform.invocation.selectors:
        if normalized == normalize_identifier(selector.model_id):
            return selector
        if text.startswith(f"{selector.name.casefold()}."):
            return selector
    return None


def derive_invocation(
    platform: PlatformAvailability,
    spelling: str | None,
) -> InvocationDerivation:
    """Derive the invocation identity for one resolved platform representation.

    Fail-open semantics: without reviewed invocation facts the bare model id
    is used exactly as before (with a warning); only a profile that states
    `bare_on_demand_supported: false` requires a selector, and then the
    user's spelling seeds the choice, a sole selector is chosen with a note,
    and multiple selectors require an explicit user selection.
    """
    invocation = platform.invocation
    spelled = selector_for_spelling(platform, spelling)
    if spelled is not None:
        return InvocationDerivation(
            choice=InvocationChoice(
                invocation_model_id=spelled.model_id,
                selector=spelled.name,
                requires_selector=invocation is not None
                and invocation.bare_on_demand_supported is False,
                notes=[
                    f"Invocation selector {spelled.name!r} "
                    f"({spelled.model_id}) selected from the original spelling "
                    f"{spelling!r}."
                ],
            )
        )
    if invocation is None:
        return InvocationDerivation(
            choice=InvocationChoice(
                invocation_model_id=platform.model_id,
                warnings=[
                    "No reviewed invocation facts exist for "
                    f"{platform.platform}/{platform.model_id}; the bare platform "
                    "model id is used as the invocation id. Verify it is "
                    "invocable on demand before deploying."
                ],
            )
        )
    if invocation.bare_on_demand_supported is False:
        if len(invocation.selectors) == 1:
            only = invocation.selectors[0]
            return InvocationDerivation(
                choice=InvocationChoice(
                    invocation_model_id=only.model_id,
                    selector=only.name,
                    requires_selector=True,
                    notes=[
                        f"Selected the only invocation selector {only.name!r} "
                        f"({only.model_id}); the bare id "
                        f"{platform.model_id!r} is not invocable on demand."
                    ],
                )
            )
        return InvocationDerivation(selection_required=list(invocation.selectors))
    warnings = (
        []
        if invocation.bare_on_demand_supported
        else [
            "The reviewed invocation facts for "
            f"{platform.platform}/{platform.model_id} do not state whether the "
            "bare model id is invocable on demand; it is used as the invocation "
            "id. Verify before deploying."
        ]
    )
    return InvocationDerivation(
        choice=InvocationChoice(invocation_model_id=platform.model_id, warnings=warnings)
    )


def platform_spelling_set(platform: PlatformAvailability) -> list[str]:
    """Every reviewed spelling that names this representation: bare + selectors."""
    spellings = [platform.model_id]
    if platform.invocation is not None:
        spellings.extend(
            selector.model_id
            for selector in platform.invocation.selectors
            if selector.model_id not in spellings
        )
    return spellings


def qualify_bare_references(
    text: str, bare_id: str, qualified_ids: list[str], replacement: str
) -> str:
    """Replace bare-id references with the invocation id, keeping qualified ones.

    Occurrences of the bare id already inside a selector-qualified spelling are
    left untouched; standalone occurrences become `replacement`.
    """
    if bare_id == replacement or bare_id not in text:
        return text
    covering = [
        (qualified, qualified.find(bare_id))
        for qualified in qualified_ids
        if bare_id in qualified and qualified != bare_id
    ]
    parts: list[str] = []
    position = 0
    while True:
        index = text.find(bare_id, position)
        if index == -1:
            parts.append(text[position:])
            return "".join(parts)
        covered = any(
            text[index - offset : index - offset + len(qualified)] == qualified
            for qualified, offset in covering
            if index - offset >= 0
        )
        parts.append(text[position:index])
        parts.append(bare_id if covered else replacement)
        position = index + len(bare_id)


def references_bare_alone(text: str, bare_id: str, qualified_ids: list[str]) -> bool:
    """Whether `text` references `bare_id` outside every selector-qualified form.

    Each occurrence of the bare id is checked against the qualified ids that
    contain it; an occurrence not covered by any of them is a bare reference.
    """
    covering = [
        (qualified, qualified.find(bare_id))
        for qualified in qualified_ids
        if bare_id in qualified and qualified != bare_id
    ]
    start = 0
    while True:
        index = text.find(bare_id, start)
        if index == -1:
            return False
        covered = any(
            text[index - offset : index - offset + len(qualified)] == qualified
            for qualified, offset in covering
            if index - offset >= 0
        )
        if not covered:
            return True
        start = index + 1
