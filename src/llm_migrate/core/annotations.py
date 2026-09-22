"""Hunk-anchored, evidence-linked change annotations for adaptation deliverables.

Every meaningful edit in a submitted deliverable must be recorded as one
:class:`AnnotatedChange`: an anchored span in the original and/or adapted
content, a one-sentence `why`, and the evidence behind it. Validation is
deterministic and fail-closed against the real diff of the DECODED runtime
values: every diff hunk must be covered by an annotation whose anchors
resolve (no undocumented edits), and every annotation must correspond to a
real hunk (no phantom claims). The annotations live beside the deliverable in
`changes.yaml`; the deliverable itself stays clean.
"""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from typing import Literal

from pydantic import Field

from llm_migrate.core.models import (
    MigrationPlan,
    PromptSourceFormat,
    StrictModel,
)
from llm_migrate.core.prompt_documents import (
    PromptDocumentError,
    extract_prompt_components,
    parse_structured_document,
)

ChangeOperation = Literal["edit", "insert", "delete", "restructure"]
EvidenceKind = Literal[
    "model_guidance", "model_difference", "analysis_finding", "research", "mechanical"
]


class ChangeEvidence(StrictModel):
    """One evidence entry backing an annotated change.

    `mechanical` covers typo-level fixes that need no citation; every other
    kind must name its source through `url` (a link the run's plan already
    carries) or `reference` (a registry field, report section, or finding
    category).
    """

    kind: EvidenceKind
    url: str = ""
    reference: str = ""


class AnnotatedChange(StrictModel):
    """One anchored, explained, evidence-linked change in a deliverable.

    Anchors are exact text spans of the DECODED runtime content: an `edit`
    names both the original and the adapted span, a `delete` only the
    original, an `insert` only the adapted, and a `restructure` with no
    anchors claims the whole rewrite (only meaningful together with an
    acknowledged restructure). The `id` is content-derived at acceptance, so
    a recorded review decision keeps matching the same change.
    """

    id: str = ""
    operation: ChangeOperation
    original_anchor: str | None = None
    adapted_anchor: str | None = None
    why: str
    evidence: list[ChangeEvidence] = Field(default_factory=list)


def change_id(change: AnnotatedChange) -> str:
    """Deterministic stable id over the change's content (blocker-id scheme)."""
    digest = hashlib.sha256(
        "|".join(
            (
                change.operation,
                change.original_anchor or "",
                change.adapted_anchor or "",
                change.why,
            )
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"c:{digest}"


def with_change_ids(changes: list[AnnotatedChange]) -> list[AnnotatedChange]:
    """The same annotations with their deterministic ids filled in."""
    return [item.model_copy(update={"id": change_id(item)}) for item in changes]


def decoded_view(text: str, format: PromptSourceFormat | None) -> str | None:
    """The DECODED runtime view of a deliverable, for diffing and anchoring.

    Structured prompt documents decode to their prompt component values
    (named components sorted by key so serialization order carries no
    meaning, then positional message content in order); plain text and
    non-prompt files are their own view. None means the document could not
    be parsed, and annotation checks must be skipped and disclosed, never
    silently passed.
    """
    if format is None:
        return text
    try:
        components = extract_prompt_components(parse_structured_document(text, format))
    except PromptDocumentError:
        return None
    named = sorted(
        (component.key, component.content)
        for component in components
        if component.key and "[" not in component.key
    )
    positional = [
        component.content for component in components if component.key and "[" in component.key
    ]
    return "\n\n".join([*(content for _, content in named), *positional])


def _line_offsets(lines: list[str]) -> list[int]:
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def _anchor_spans(text: str, anchor: str) -> list[tuple[int, int]]:
    """Character intervals of every non-overlapping occurrence of the anchor."""
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        found = text.find(anchor, start)
        if found < 0:
            return spans
        spans.append((found, found + len(anchor)))
        start = found + len(anchor)


def _overlaps(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(span_start < end and start < span_end for span_start, span_end in spans)


def _shorten(text: str, limit: int = 100) -> str:
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[: limit - 1] + "…"


def _hunks(original: str, adapted: str) -> list[tuple[str, int, int, int, int]]:
    """Non-equal diff hunks as (tag, orig_start, orig_end, new_start, new_end)
    character intervals, computed line-wise over the decoded views."""
    original_lines = original.splitlines(keepends=True)
    adapted_lines = adapted.splitlines(keepends=True)
    original_offsets = _line_offsets(original_lines)
    adapted_offsets = _line_offsets(adapted_lines)
    matcher = SequenceMatcher(None, original_lines, adapted_lines, autojunk=False)
    return [
        (
            tag,
            original_offsets[i1],
            original_offsets[i2],
            adapted_offsets[j1],
            adapted_offsets[j2],
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _label(change: AnnotatedChange) -> str:
    return f"{change.operation} ({_shorten(change.why, 60)})"


def _shape_problems(change: AnnotatedChange) -> list[str]:
    """Operation/anchor consistency for one annotation."""
    problems: list[str] = []
    original = (change.original_anchor or "").strip()
    adapted = (change.adapted_anchor or "").strip()
    if not change.why.strip():
        problems.append(f"annotated change {_label(change)} has an empty `why`")
    if change.operation == "edit" and not (original and adapted):
        problems.append(
            f"annotated change {_label(change)}: an edit requires both "
            "original_anchor and adapted_anchor"
        )
    if change.operation == "delete":
        if not original:
            problems.append(f"annotated change {_label(change)}: a delete requires original_anchor")
        if adapted:
            problems.append(
                f"annotated change {_label(change)}: a delete cannot carry adapted_anchor"
            )
    if change.operation == "insert":
        if not adapted:
            problems.append(f"annotated change {_label(change)}: an insert requires adapted_anchor")
        if original:
            problems.append(
                f"annotated change {_label(change)}: an insert cannot carry original_anchor"
            )
    return problems


def _evidence_problems(
    change: AnnotatedChange, known_urls: set[str], strict_evidence: bool = False
) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    if not change.evidence:
        problems.append(
            f"annotated change {_label(change)} carries no evidence entry; cite the "
            "guidance, model difference, finding, or research behind it, or record "
            "kind 'mechanical' for a typo-level fix"
        )
    for entry in change.evidence:
        if entry.kind == "mechanical":
            continue
        if not entry.url.strip() and not entry.reference.strip():
            problems.append(
                f"annotated change {_label(change)}: evidence of kind {entry.kind!r} "
                "needs a url or a reference"
            )
        elif entry.url.strip() and known_urls and entry.url not in known_urls:
            message = (
                f"annotated change {_label(change)}: evidence URL {entry.url} is not "
                "among the run's known evidence sources"
            )
            if strict_evidence:
                problems.append(message + " (strict mode: unknown evidence URLs are rejected)")
            else:
                warnings.append(message)
    return problems, warnings


def _duplicate_problems(changes: list[AnnotatedChange]) -> list[str]:
    """Identical annotations would share one content-derived id; refuse them."""
    counts: dict[str, int] = {}
    for change in changes:
        identity = change_id(change)
        counts[identity] = counts.get(identity, 0) + 1
    return [
        f"{count} identical annotated changes would share the id {identity}; "
        "merge them into one entry"
        for identity, count in sorted(counts.items())
        if count > 1
    ]


def standalone_annotation_problems(
    changes: list[AnnotatedChange],
    known_evidence_urls: set[str] | None = None,
    strict_evidence: bool = False,
) -> tuple[list[str], list[str]]:
    """Shape, evidence, and duplicate checks without a diff to reconcile against.

    Used for deliverables that have no original to diff (new files): anchors
    cannot be resolved, but the why/evidence/operation rules still hold.
    """
    problems: list[str] = list(_duplicate_problems(changes))
    warnings: list[str] = []
    known_urls = known_evidence_urls or set()
    for change in changes:
        problems.extend(_shape_problems(change))
        evidence_problems, evidence_warnings = _evidence_problems(
            change, known_urls, strict_evidence
        )
        problems.extend(evidence_problems)
        warnings.extend(evidence_warnings)
    return problems, warnings


def annotation_problems(
    original: str,
    adapted: str,
    changes: list[AnnotatedChange],
    known_evidence_urls: set[str] | None = None,
    strict_evidence: bool = False,
) -> tuple[list[str], list[str]]:
    """Fail-closed reconciliation of annotations against the real diff.

    Returns (problems, warnings) over the decoded views: shape and evidence
    checks per annotation, anchors that must resolve in their text, every
    annotation matched to at least one hunk, and every hunk covered by at
    least one annotation.
    """
    problems: list[str] = []
    warnings: list[str] = []
    known_urls = known_evidence_urls or set()
    problems.extend(_duplicate_problems(changes))
    hunks = _hunks(original, adapted)
    covered = [False] * len(hunks)
    global_restructure = False
    for change in changes:
        problems.extend(_shape_problems(change))
        evidence_problems, evidence_warnings = _evidence_problems(
            change, known_urls, strict_evidence
        )
        problems.extend(evidence_problems)
        warnings.extend(evidence_warnings)
        original_anchor = (change.original_anchor or "").strip()
        adapted_anchor = (change.adapted_anchor or "").strip()
        if change.operation == "restructure" and not original_anchor and not adapted_anchor:
            # A global restructure claims the whole rewrite.
            covered = [True] * len(hunks)
            global_restructure = True
            continue
        original_spans = _anchor_spans(original, original_anchor) if original_anchor else []
        adapted_spans = _anchor_spans(adapted, adapted_anchor) if adapted_anchor else []
        if original_anchor and not original_spans:
            problems.append(
                f"annotated change {_label(change)}: original_anchor "
                f"{_shorten(original_anchor)!r} does not occur in the original content"
            )
            continue
        if adapted_anchor and not adapted_spans:
            problems.append(
                f"annotated change {_label(change)}: adapted_anchor "
                f"{_shorten(adapted_anchor)!r} does not occur in the adapted content"
            )
            continue
        matched = False
        for index, (_, orig_start, orig_end, new_start, new_end) in enumerate(hunks):
            if _overlaps(original_spans, orig_start, orig_end) or _overlaps(
                adapted_spans, new_start, new_end
            ):
                covered[index] = True
                matched = True
        if not matched:
            problems.append(
                f"annotated change {_label(change)} does not correspond to any actual "
                "change between the original and the adapted content"
            )
    for index, (tag, orig_start, orig_end, new_start, new_end) in enumerate(hunks):
        if covered[index]:
            continue
        removed = original[orig_start:orig_end]
        added = adapted[new_start:new_end]
        detail = "; ".join(
            part
            for part in (
                f"removed: {_shorten(removed)!r}" if removed.strip() else "",
                f"added: {_shorten(added)!r}" if added.strip() else "",
            )
            if part
        )
        problems.append(
            f"undocumented {tag} change with no covering annotation ({detail}); every "
            "meaningful edit needs an annotated change with its why and evidence"
        )
    if changes and not hunks and not global_restructure:
        # No diff at all: every provided annotation is a phantom, already
        # reported above per annotation; nothing more to add.
        pass
    return problems, warnings


def plan_evidence_urls(plan: MigrationPlan) -> set[str]:
    """Every evidence URL the run's plan already carries, for honesty checks."""
    urls: set[str] = set()
    for spec in plan.prompt_changes:
        for bucket in (
            spec.requirements_to_preserve,
            spec.assumptions_to_reconsider,
            spec.instructions_potentially_removable,
            spec.instructions_needing_strengthening,
            spec.target_features_replacing_prompt_text,
            spec.reasoning_configuration,
            spec.structured_output,
            spec.token_efficiency_opportunities,
            spec.hallucination_control,
            spec.migration_risks,
            spec.validation_recommendations,
        ):
            for advice in bucket:
                urls.update(advice.evidence_urls)
    for difference in plan.model_differences.differences:
        for url in (difference.source_evidence_url, difference.target_evidence_url):
            if url:
                urls.add(url)
        urls.update(source for source in difference.supporting_sources if source.startswith("http"))
    for blocker in plan.blockers:
        urls.update(blocker.evidence_urls)
    return urls


def apply_decided_changes(
    original: str,
    adapted: str,
    changes: list[AnnotatedChange],
    rejected_ids: set[str],
) -> tuple[str | None, list[str], set[str]]:
    """The adapted content with every rejected change reverted, hunk by hunk.

    Deterministic replay of the submission diff: an edited region covered
    only by rejected changes takes the original side, every other edited
    region keeps the adapted side (pending changes stand until rejected).
    Returns (content, problems, mapped_rejected_ids); content is None when a
    region is covered by both a rejected and a non-rejected change, which
    cannot be separated deterministically. `mapped_rejected_ids` are the
    rejected ids that covered at least one region here — the caller fails
    closed when a rejected id maps onto nothing, rather than letting a
    rejection silently change nothing.
    """
    original_lines = original.splitlines(keepends=True)
    adapted_lines = adapted.splitlines(keepends=True)
    original_offsets = _line_offsets(original_lines)
    adapted_offsets = _line_offsets(adapted_lines)
    resolved: list[tuple[AnnotatedChange, list[tuple[int, int]], list[tuple[int, int]], bool]] = []
    for change in changes:
        original_anchor = (change.original_anchor or "").strip()
        adapted_anchor = (change.adapted_anchor or "").strip()
        global_restructure = (
            change.operation == "restructure" and not original_anchor and not adapted_anchor
        )
        resolved.append(
            (
                change,
                _anchor_spans(original, original_anchor) if original_anchor else [],
                _anchor_spans(adapted, adapted_anchor) if adapted_anchor else [],
                global_restructure,
            )
        )
    problems: list[str] = []
    mapped_rejected: set[str] = set()
    parts: list[str] = []
    matcher = SequenceMatcher(None, original_lines, adapted_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append("".join(original_lines[i1:i2]))
            continue
        orig_start, orig_end = original_offsets[i1], original_offsets[i2]
        new_start, new_end = adapted_offsets[j1], adapted_offsets[j2]
        rejected_here: set[str] = set()
        kept_here: set[str] = set()
        for change, original_spans, adapted_spans, global_restructure in resolved:
            covers = (
                global_restructure
                or _overlaps(original_spans, orig_start, orig_end)
                or _overlaps(adapted_spans, new_start, new_end)
            )
            if not covers:
                continue
            if change.id in rejected_ids:
                rejected_here.add(change.id)
            else:
                kept_here.add(change.id)
        if rejected_here and kept_here:
            problems.append(
                "one edited region is covered by both rejected ("
                + ", ".join(sorted(rejected_here))
                + ") and non-rejected ("
                + ", ".join(sorted(kept_here))
                + ") changes; the decisions cannot be applied deterministically — "
                "resubmit an adapted version that reflects them instead"
            )
        mapped_rejected.update(rejected_here)
        if rejected_here:
            parts.append("".join(original_lines[i1:i2]))
        else:
            parts.append("".join(adapted_lines[j1:j2]))
    if problems:
        return None, problems, mapped_rejected
    return "".join(parts), [], mapped_rejected
