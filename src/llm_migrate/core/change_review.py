"""Per-change review of adaptation deliverables: present, decide, regenerate.

The review surface pairs every submitted deliverable's annotated changes with
their decision state so a user can accept or reject each change individually,
the way a pull request is reviewed. Decisions are durable in the run's
``change-decisions.yaml``, keyed to the submission's content fingerprints
(a resubmission makes them stale, never silently applied), and every recorded
decision deterministically regenerates the deliverable under ``output/`` from
the original content, the as-submitted content, and the rejected change ids —
rejecting every change reverts the deliverable to carrying no adaptation.
The toolkit never decides; it derives, records, and replays.
"""

from __future__ import annotations

import hashlib
from datetime import date
from difflib import unified_diff
from pathlib import Path
from typing import Literal

from pydantic import Field

from llm_migrate.core.annotations import (
    AnnotatedChange,
    apply_decided_changes,
    decoded_view,
)
from llm_migrate.core.models import StrictModel
from llm_migrate.core.prompt_documents import (
    STRUCTURED_SUFFIXES,
    PromptDocumentError,
    build_candidate_document,
    extract_prompt_components,
    parse_structured_document,
)
from llm_migrate.core.runstate import atomic_write_text, run_state_lock
from llm_migrate.core.workspace import (
    AdaptationEntry,
    ChangeDecision,
    MigrationRunConfig,
    decision_matches_entry,
    entry_is_unchanged,
    load_adaptation_log,
    load_change_decision_log,
    read_original_prompt,
    save_change_decision_log,
    submission_copy_path,
    upsert_change_decision,
)

CHANGE_REVIEW_GUIDANCE = [
    "Present each pending change VERBATIM — its why, evidence, and before/after "
    "spans — one change at a time; the user accepts or rejects each change, "
    "never you.",
    "Record each user decision with record_change_decision "
    "(CLI: `llm-migrate run decide-change`). Rejecting a change "
    "deterministically regenerates the deliverable from the remaining changes; "
    "rejecting every change leaves no annotated adaptation in the deliverable.",
    "Decisions persist in change-decisions.yaml and are keyed to the submitted "
    "content: resubmitting a file makes its previous decisions stale (reported, "
    "never silently applied).",
    "A review marked stale (the application file drifted, or the as-submitted "
    "content is unavailable) cannot take decisions; re-run the adaptation for "
    "that file first.",
]


class ChangeReviewItem(StrictModel):
    """One annotated change paired with its current decision state."""

    change: AnnotatedChange
    status: Literal["pending", "accepted", "rejected"]
    note: str = ""
    decided_on: date | None = None


class FileChangeReview(StrictModel):
    """Everything needed to review one deliverable change by change.

    `diff` is the unified diff of the DECODED original versus the DECODED
    as-submitted content — what actually changed at runtime, independent of
    serialization. `stale_reason` is set when decisions cannot be taken.
    """

    source_path: str
    kind: Literal["prompt", "file"]
    output_path: str
    unchanged: bool = False
    stale_reason: str | None = None
    diff: str = ""
    changes: list[ChangeReviewItem] = Field(default_factory=list)
    pending: int = 0


class ChangeReviewSet(StrictModel):
    """The whole run's per-change review state, for verbatim presentation."""

    schema_version: Literal["1"] = "1"
    run_id: str
    files: list[FileChangeReview] = Field(default_factory=list)
    stale_decisions: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


class ChangeDecisionResult(StrictModel):
    """Outcome of recording one change decision, with the refreshed state."""

    accepted: bool
    source_path: str = ""
    change_id: str = ""
    problems: list[str] = Field(default_factory=list)
    pending_change_ids: list[str] = Field(default_factory=list)
    deliverable_path: str | None = None
    reverted_to_original: bool = False
    message: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entry_format(entry: AdaptationEntry):  # type: ignore[no-untyped-def]
    if entry.kind != "prompt":
        return None
    return STRUCTURED_SUFFIXES.get(Path(entry.source_path).suffix.casefold())


def _submission_content(run_dir: Path, entry: AdaptationEntry) -> str | None:
    """The as-submitted content of one deliverable, verified by fingerprint.

    Falls back to the output file for pre-review submissions, but only while
    it still matches the submission fingerprint (a regenerated or edited
    output file is not the submission).
    """
    copy = submission_copy_path(run_dir, entry.kind, entry.source_path)
    candidates = [copy, Path(run_dir) / entry.output_path]
    for path in candidates:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if _sha256(content) == entry.adapted_sha256:
            return content
    return None


def _entry_staleness(
    run_dir: Path, config: MigrationRunConfig, entry: AdaptationEntry
) -> tuple[str | None, str | None, str | None]:
    """(original, submission, stale_reason) for one reviewable entry."""
    original = read_original_prompt(config, entry.source_path)
    if entry.source_sha256 is None:
        return (
            None,
            None,
            ("the submission recorded no original fingerprint; per-change review does not apply"),
        )
    if original is None:
        return None, None, "the original file is no longer present in the application"
    if _sha256(original) != entry.source_sha256:
        return (
            None,
            None,
            (
                "the application file changed since submission; decisions are frozen "
                "until the adaptation is re-run"
            ),
        )
    submission = _submission_content(run_dir, entry)
    if submission is None:
        return (
            original,
            None,
            (
                "the as-submitted content is unavailable; resubmit the adaptation to "
                "enable per-change review"
            ),
        )
    return original, submission, None


def _decoded_diff(entry: AdaptationEntry, original: str, submission: str) -> str:
    format = _entry_format(entry)
    decoded_original = decoded_view(original, format)
    decoded_submission = decoded_view(submission, format)
    if decoded_original is None or decoded_submission is None:
        return ""
    return "".join(
        unified_diff(
            decoded_original.splitlines(keepends=True),
            decoded_submission.splitlines(keepends=True),
            fromfile=f"a/{entry.source_path}",
            tofile=f"b/{entry.source_path}",
        )
    )


def build_change_review(run_dir: Path, config: MigrationRunConfig) -> ChangeReviewSet:
    """Every deliverable's annotated changes paired with their decision state."""
    log = load_adaptation_log(run_dir, config.run_id)
    decision_log = load_change_decision_log(run_dir, config.run_id)
    files: list[FileChangeReview] = []
    live_decision_keys: set[tuple[str, str]] = set()
    for entry in sorted(log.entries, key=lambda item: (item.kind, item.source_path)):
        decisions = {
            decision.change_id: decision
            for decision in decision_log.decisions
            if decision_matches_entry(decision, entry)
        }
        live_decision_keys.update(
            (decision.source_path, decision.change_id) for decision in decisions.values()
        )
        if entry_is_unchanged(entry):
            files.append(
                FileChangeReview(
                    source_path=entry.source_path,
                    kind=entry.kind,
                    output_path=entry.output_path,
                    unchanged=True,
                )
            )
            continue
        original, submission, stale_reason = _entry_staleness(run_dir, config, entry)
        if stale_reason is None and not entry.annotated_changes:
            stale_reason = (
                "the submission carries no annotated changes (submitted before "
                "per-change review); resubmit the adaptation to enable review"
            )
        items = [
            ChangeReviewItem(
                change=change,
                status=(decisions[change.id].decision if change.id in decisions else "pending"),
                note=decisions[change.id].note if change.id in decisions else "",
                decided_on=(decisions[change.id].decided_on if change.id in decisions else None),
            )
            for change in entry.annotated_changes
        ]
        files.append(
            FileChangeReview(
                source_path=entry.source_path,
                kind=entry.kind,
                output_path=entry.output_path,
                stale_reason=stale_reason,
                diff=(
                    _decoded_diff(entry, original, submission)
                    if original is not None and submission is not None
                    else ""
                ),
                changes=items,
                pending=sum(item.status == "pending" for item in items),
            )
        )
    stale_decisions = [
        f"{decision.source_path} [{decision.change_id}]: {decision.decision} decision "
        f"recorded on {decision.decided_on.isoformat()} no longer matches the current "
        "submission and was NOT applied"
        for decision in decision_log.decisions
        if (decision.source_path, decision.change_id) not in live_decision_keys
    ]
    return ChangeReviewSet(
        run_id=config.run_id,
        files=files,
        stale_decisions=stale_decisions,
        guidance=list(CHANGE_REVIEW_GUIDANCE),
    )


def _regenerate_content(
    entry: AdaptationEntry,
    original: str,
    submission: str,
    rejected_ids: set[str],
) -> tuple[str | None, list[str]]:
    """The effective deliverable content under the given rejections."""
    if not rejected_ids:
        return submission, []
    changes = list(entry.annotated_changes)
    format = _entry_format(entry)
    if format is None:
        content, plain_problems, plain_mapped = apply_decided_changes(
            original, submission, changes, rejected_ids
        )
        plain_unmapped = rejected_ids - plain_mapped
        if plain_unmapped and not plain_problems:
            plain_problems = [
                "rejected change(s) "
                + ", ".join(sorted(plain_unmapped))
                + " do not map onto any edited region; resubmit an adapted version "
                "that reflects the decisions instead"
            ]
        return (None, plain_problems) if plain_problems else (content, [])
    try:
        original_doc = parse_structured_document(original, format)
        submission_doc = parse_structured_document(submission, format)
    except PromptDocumentError as exc:
        return None, [f"the document could not be parsed for regeneration: {exc}"]
    original_components = extract_prompt_components(original_doc)
    submission_components = extract_prompt_components(submission_doc)
    original_named = {
        component.key: component.content
        for component in original_components
        if component.key and "[" not in component.key
    }
    submission_named = {
        component.key: component.content
        for component in submission_components
        if component.key and "[" not in component.key
    }
    original_positional = [
        component.content
        for component in original_components
        if component.key and "[" in component.key
    ]
    submission_positional = [
        component.content
        for component in submission_components
        if component.key and "[" in component.key
    ]
    if set(original_named) != set(submission_named) or original_positional != submission_positional:
        return None, [
            "the submission adds, removes, or edits message components; rejections "
            "cannot be applied to this document deterministically — resubmit an "
            "adapted version that reflects the decisions instead"
        ]
    problems: list[str] = []
    mapped: set[str] = set()
    replacements: dict[str, str] = {}
    for key, submitted_value in submission_named.items():
        original_value = original_named[key]
        if original_value == submitted_value:
            continue
        value, value_problems, value_mapped = apply_decided_changes(
            original_value, submitted_value, changes, rejected_ids
        )
        problems.extend(value_problems)
        mapped |= value_mapped
        if value is not None:
            replacements[key] = value
    unmapped = rejected_ids - mapped
    if unmapped and not problems:
        problems.append(
            "rejected change(s) "
            + ", ".join(sorted(unmapped))
            + " do not map onto any edited region of this document; resubmit an "
            "adapted version that reflects the decisions instead"
        )
    if problems:
        return None, problems
    rebuilt = build_candidate_document(submission, format, replacements)
    if rebuilt is None:
        return None, [
            f"{format.value} documents cannot be deterministically rebuilt; resubmit "
            "an adapted version that reflects the decisions instead"
        ]
    return rebuilt, []


def record_change_decision(
    run_dir: Path,
    config: MigrationRunConfig,
    source_path: str,
    change_id: str,
    decision: str,
    note: str = "",
    decided_on: date | None = None,
) -> ChangeDecisionResult:
    """Record one accept/reject decision and regenerate the deliverable.

    Fail-closed: the decision is saved only when the resulting deliverable can
    be regenerated deterministically from the original content, the
    as-submitted content, and every live rejection.
    """
    log = load_adaptation_log(run_dir, config.run_id)
    entries = [entry for entry in log.entries if entry.source_path == source_path]
    if not entries:
        known = ", ".join(sorted({entry.source_path for entry in log.entries})) or "(none)"
        return ChangeDecisionResult(
            accepted=False,
            source_path=source_path,
            change_id=change_id,
            problems=[f"no deliverable was submitted for {source_path!r}; known paths: {known}"],
            message="Rejected: unknown deliverable path.",
        )
    if len(entries) > 1:
        return ChangeDecisionResult(
            accepted=False,
            source_path=source_path,
            change_id=change_id,
            problems=[
                f"{source_path!r} has both a prompt and a file deliverable; resolve the "
                "duplicate submission before reviewing"
            ],
            message="Rejected: ambiguous deliverable path.",
        )
    entry = entries[0]
    if decision not in ("accepted", "rejected"):
        return ChangeDecisionResult(
            accepted=False,
            source_path=source_path,
            change_id=change_id,
            problems=[f"decision must be 'accepted' or 'rejected', got {decision!r}"],
            message="Rejected: invalid decision.",
        )
    if entry_is_unchanged(entry):
        return ChangeDecisionResult(
            accepted=False,
            source_path=source_path,
            change_id=change_id,
            problems=["this deliverable was reviewed as unchanged; it has no changes to decide"],
            message="Rejected: nothing to decide on an unchanged deliverable.",
        )
    if all(change.id != change_id for change in entry.annotated_changes):
        valid = ", ".join(change.id for change in entry.annotated_changes) or "(none)"
        return ChangeDecisionResult(
            accepted=False,
            source_path=source_path,
            change_id=change_id,
            problems=[f"unknown change id {change_id!r}; this deliverable's changes: {valid}"],
            message="Rejected: unknown change id.",
        )
    original, submission, stale_reason = _entry_staleness(Path(run_dir), config, entry)
    if stale_reason is not None or original is None or submission is None:
        return ChangeDecisionResult(
            accepted=False,
            source_path=source_path,
            change_id=change_id,
            problems=[stale_reason or "the review state for this file is unavailable"],
            message="Rejected: " + (stale_reason or "review unavailable."),
        )
    with run_state_lock(run_dir):
        decision_log = load_change_decision_log(run_dir, config.run_id)
        candidate = ChangeDecision(
            source_path=entry.source_path,
            kind=entry.kind,
            change_id=change_id,
            decision=decision,  # type: ignore[arg-type]
            note=note,
            decided_on=decided_on or date.today(),
            source_sha256=entry.source_sha256,
            adapted_sha256=entry.adapted_sha256,
        )
        updated_log = upsert_change_decision(decision_log, candidate)
        rejected_ids = {
            item.change_id
            for item in updated_log.decisions
            if item.decision == "rejected" and decision_matches_entry(item, entry)
        }
        content, problems = _regenerate_content(entry, original, submission, rejected_ids)
        if problems or content is None:
            return ChangeDecisionResult(
                accepted=False,
                source_path=source_path,
                change_id=change_id,
                problems=problems,
                message="Rejected (decision NOT recorded): " + "; ".join(problems),
            )
        save_change_decision_log(run_dir, updated_log)
        deliverable = Path(run_dir) / entry.output_path
        atomic_write_text(deliverable, content)
    decided = {
        item.change_id for item in updated_log.decisions if decision_matches_entry(item, entry)
    }
    pending = [change.id for change in entry.annotated_changes if change.id not in decided]
    reverted = content == original
    message = (
        f"Recorded {decision} for {change_id} on {entry.source_path}; the deliverable "
        f"was regenerated ({len(rejected_ids)} rejection(s) applied)."
    )
    if reverted:
        message += " Every change is rejected: the deliverable is the original content."
    if pending:
        message += f" {len(pending)} change(s) still pending review."
    return ChangeDecisionResult(
        accepted=True,
        source_path=entry.source_path,
        change_id=change_id,
        pending_change_ids=pending,
        deliverable_path=str(deliverable),
        reverted_to_original=reverted,
        message=message,
    )


def reapply_change_decisions(
    run_dir: Path,
    config: MigrationRunConfig,
    source_path: str,
) -> tuple[int, list[str]]:
    """Re-apply live rejections to one deliverable after a (re)submission.

    A resubmission whose content fingerprints match the previous submission
    keeps its recorded decisions live, so the deliverable under output/ must
    keep reflecting them instead of silently reverting to the full
    submission. Returns (applied_rejections, problems).
    """
    with run_state_lock(run_dir):
        log = load_adaptation_log(run_dir, config.run_id)
        entries = [entry for entry in log.entries if entry.source_path == source_path]
        if len(entries) != 1 or entry_is_unchanged(entries[0]):
            return 0, []
        entry = entries[0]
        decision_log = load_change_decision_log(run_dir, config.run_id)
        rejected_ids = {
            item.change_id
            for item in decision_log.decisions
            if item.decision == "rejected" and decision_matches_entry(item, entry)
        }
        if not rejected_ids:
            return 0, []
        original, submission, stale_reason = _entry_staleness(Path(run_dir), config, entry)
        if stale_reason is not None or original is None or submission is None:
            return 0, [stale_reason or "the review state for this file is unavailable"]
        content, problems = _regenerate_content(entry, original, submission, rejected_ids)
        if problems or content is None:
            return 0, problems
        deliverable = Path(run_dir) / entry.output_path
        atomic_write_text(deliverable, content)
        return len(rejected_ids), []
