"""Cross-surface consistency gate over the finalized deliverable set.

Individual submissions are checked one at a time; nothing before this gate
verifies that the deliverables agree WITH EACH OTHER: that invocation,
configuration, telemetry attribution, and pricing all reference the same
selector-qualified target, that no active source-model configuration remains
anywhere (including deliverables reverted by review rejections), and that a
coupling the scanner recognized in the source did not silently evaporate from
the adapted content. The gate is deterministic and honestly bounded: it
checks the coupling kinds the scanner recognizes through their lexical
markers, no more. Violations are report findings by default; strict mode
turns them into blockers.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field

from llm_migrate.core.invocation_identity import references_bare_alone
from llm_migrate.core.models import ApplicationAnalysis, CouplingKind, ModelPricing, StrictModel
from llm_migrate.core.pricing_gate import classify_price
from llm_migrate.core.prompt_documents import (
    PromptDocumentError,
    parse_structured_document,
    source_format,
)
from llm_migrate.core.workspace import (
    AdaptationEntry,
    AdaptationLog,
    MigrationRunConfig,
    effective_target_id,
    entry_is_unchanged,
    source_detection_spellings,
    source_reference_spellings,
    target_reference_spellings,
    target_spellings,
)
from llm_migrate.scanners.python import scannable_files

ConsistencyCode = Literal[
    "source_reference_remains",
    "bare_target_reference",
    "mixed_target_selectors",
    "target_reference_missing",
    "dropped_coupling_undisposed",
    "source_reference_uncovered",
    "pricing_contradicts_registry",
    "pricing_stale_source",
    "pricing_unit_unrecognized",
]


class ConsistencyFinding(StrictModel):
    """One deterministic cross-surface disagreement in the deliverable set."""

    code: ConsistencyCode
    path: str
    message: str

    @property
    def rendered(self) -> str:
        return f"[{self.code}] {self.path}: {self.message}"


# Lexical marker per recognized coupling kind. A marker must be a distinctive
# spelling the scanner itself matched on; kinds without a reliable marker are
# deliberately not checked (bounded honesty).
_MARKER_KINDS = {
    CouplingKind.PARAMETER,
    CouplingKind.TOOL,
    CouplingKind.STRUCTURED_OUTPUT,
}
_GENERIC_MARKERS = {"text", "prompt", "input", "stream"}


def _coupling_markers(analysis: ApplicationAnalysis) -> dict[str, dict[str, str]]:
    """Per source file: lexical marker -> the coupling it stands for."""
    markers: dict[str, dict[str, str]] = {}
    for finding in analysis.findings:
        if finding.kind not in _MARKER_KINDS:
            continue
        name = (finding.metadata or {}).get("name")
        marker = name if isinstance(name, str) else finding.value
        if not isinstance(marker, str) or not marker or marker.casefold() in _GENERIC_MARKERS:
            continue
        markers.setdefault(finding.location.path, {}).setdefault(
            marker, f"{finding.kind.value} coupling ({finding.detail})"
        )
    return markers


def _effective_content(run_dir: Path, entry: AdaptationEntry) -> str | None:
    """The deliverable as it stands now (review rejections regenerate it)."""
    path = Path(run_dir) / entry.output_path
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _original_content(config: MigrationRunConfig, entry: AdaptationEntry) -> str | None:
    root = Path(config.application_root)
    base = root if root.is_dir() else root.parent
    path = base / entry.source_path
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _disposed(entry: AdaptationEntry, marker: str) -> bool:
    """Whether the entry explicitly accounts for dropping this marker."""
    for change in entry.annotated_changes:
        if marker in (change.original_anchor or "") or marker in (change.adapted_anchor or ""):
            return True
        if marker in change.why:
            return True
    return any(marker in note for note in entry.changes) or any(
        marker in disposition.note or marker in disposition.guidance
        for disposition in entry.guidance_dispositions
    )


def _uncovered_source_references(
    config: MigrationRunConfig,
    log: AdaptationLog,
    source_forms: list[str],
    accounted_paths: set[str],
) -> list[ConsistencyFinding]:
    """Application files still naming the source model that nothing accounts for.

    The deterministic net under every discovery heuristic: whatever the
    scanner failed to trace, a file in the scanned set that references the
    source model and has no deliverable, no reviewed no-change entry, no open
    worklist task (already a coverage gap), and no out-of-scope
    classification cannot finalize silently.
    """
    application = Path(config.application_root)
    if not source_forms or not application.is_dir():
        return []
    covered = {entry.source_path for entry in log.entries} | accounted_paths
    python_files, candidate_files = scannable_files(application)
    findings: list[ConsistencyFinding] = []
    for item in sorted({*python_files, *candidate_files}):
        relative = item.relative_to(application).as_posix()
        if relative in covered:
            continue
        try:
            text = item.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        present = [spelling for spelling in source_forms if spelling in text]
        specific = [
            spelling
            for spelling in present
            if not any(spelling != other and spelling in other for other in present)
        ]
        if specific:
            findings.append(
                ConsistencyFinding(
                    code="source_reference_uncovered",
                    path=relative,
                    message=(
                        "this application file references the source model ("
                        + ", ".join(repr(item) for item in sorted(specific))
                        + ") but has no deliverable, no reviewed no-change entry, and "
                        "no worklist task; adapt it (submit_adapted_file), or — when the "
                        "reference is intentional (documentation, history) — record it "
                        "with confirm_unaffected(acknowledge_source_references=true) and "
                        "the user's own rationale"
                    ),
                )
            )
    return findings


def check_cross_surface_consistency(
    run_dir: Path,
    config: MigrationRunConfig,
    analysis: ApplicationAnalysis,
    log: AdaptationLog,
    *,
    accounted_paths: set[str] | None = None,
) -> list[ConsistencyFinding]:
    """Deterministic checks over ALL deliverables and unchanged claims together.

    `accounted_paths` are application files the run already accounts for
    without a log entry (open worklist tasks, out-of-scope files); the
    whole-application sweep skips them.
    """
    findings: list[ConsistencyFinding] = []
    target_forms = target_reference_spellings(config)
    source_forms = source_detection_spellings(config)
    selector_ids = [item for item in target_spellings(config) if item != config.target_model_id]
    markers_by_file = _coupling_markers(analysis)
    model_coupled_files = {
        finding.location.path
        for finding in analysis.findings
        if finding.kind is CouplingKind.MODEL_IDENTIFIER
    }
    selectors_in_use: dict[str, list[str]] = {}
    for entry in sorted(log.entries, key=lambda item: (item.kind, item.source_path)):
        effective = _effective_content(run_dir, entry)
        if effective is None:
            continue
        remaining = (
            []
            if entry.source_reference_acknowledged
            else [item for item in source_forms if item in effective]
        )
        for spelling in remaining:
            if any(spelling != other and spelling in other for other in remaining):
                continue  # report the most specific spelling only
            findings.append(
                ConsistencyFinding(
                    code="source_reference_remains",
                    path=entry.source_path,
                    message=(
                        f"the effective deliverable still references the source model "
                        f"id {spelling!r}; active source-model configuration must not "
                        "remain in the output set"
                    ),
                )
            )
        if (
            config.target_invocation_requires_selector
            and selector_ids
            and references_bare_alone(effective, config.target_model_id, selector_ids)
        ):
            findings.append(
                ConsistencyFinding(
                    code="bare_target_reference",
                    path=entry.source_path,
                    message=(
                        f"the effective deliverable references the bare platform model "
                        f"id {config.target_model_id!r}, which the reviewed profile "
                        "states is not invocable on demand; reference "
                        f"{effective_target_id(config)!r}"
                    ),
                )
            )
        for spelling in selector_ids:
            if spelling in effective:
                selectors_in_use.setdefault(spelling, []).append(entry.source_path)
        if (
            entry.source_path in model_coupled_files
            and not entry_is_unchanged(entry)
            and not any(item in effective for item in target_forms)
            and not remaining
        ):
            findings.append(
                ConsistencyFinding(
                    code="target_reference_missing",
                    path=entry.source_path,
                    message=(
                        "this file carried a model-identifier coupling but its adapted "
                        "deliverable references no reviewed spelling of the target "
                        f"({effective_target_id(config)!r}); model attribution "
                        "(invocation, configuration, telemetry, pricing) must name "
                        "the target"
                    ),
                )
            )
        if entry.kind == "file" and not entry_is_unchanged(entry):
            original = _original_content(config, entry)
            drifted = (
                original is None
                or entry.source_sha256 is None
                or hashlib.sha256(original.encode("utf-8")).hexdigest() != entry.source_sha256
            )
            for marker, label in sorted(markers_by_file.get(entry.source_path, {}).items()):
                if drifted or original is None or marker not in original:
                    # A drifted source belongs to the review-staleness surface,
                    # not to a marker check against content the submission
                    # never saw.
                    continue
                if marker in effective or _disposed(entry, marker):
                    continue
                findings.append(
                    ConsistencyFinding(
                        code="dropped_coupling_undisposed",
                        path=entry.source_path,
                        message=(
                            f"the source file's {label} marker {marker!r} is absent "
                            "from the adapted deliverable with no annotated change, "
                            "note, or disposition accounting for the drop"
                        ),
                    )
                )
    if len(selectors_in_use) > 1:
        detail = "; ".join(
            f"{spelling!r} in {', '.join(sorted(set(paths)))}"
            for spelling, paths in sorted(selectors_in_use.items())
        )
        findings.append(
            ConsistencyFinding(
                code="mixed_target_selectors",
                path="<run>",
                message=(
                    "the deliverable set references more than one target invocation "
                    f"selector id ({detail}); invocation, configuration, telemetry, "
                    "and pricing must agree on one selector-qualified target"
                ),
            )
        )
    findings.extend(
        _uncovered_source_references(config, log, source_forms, accounted_paths or set())
    )
    return findings


_PRICING_CODES = {
    "contradiction": "pricing_contradicts_registry",
    "stale": "pricing_stale_source",
    "unit_unrecognized": "pricing_unit_unrecognized",
}


def _lookup(data: object, key_path: tuple[str, ...]) -> object:
    node = data
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _profile_model(data: object, key_path: tuple[str, ...]) -> str | None:
    """Model id declared by the nearest enclosing mapping of a value, if any."""
    for depth in range(len(key_path) - 1, -1, -1):
        node = _lookup(data, key_path[:depth])
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            lowered = str(key).casefold()
            if isinstance(value, str) and (
                lowered in {"model", "model_id", "modelid", "model_name"}
                or lowered.endswith(("_model", "_model_id"))
            ):
                return value
    return None


def check_adapted_pricing(
    run_dir: Path,
    config: MigrationRunConfig,
    analysis: ApplicationAnalysis,
    log: AdaptationLog,
    *,
    target_pricing: ModelPricing | None,
    source_pricing: ModelPricing | None,
    known_evidence: set[str],
    as_of: date,
) -> list[ConsistencyFinding]:
    """Unit-aware registry check of every adapted value a PRICING coupling marks.

    Only configuration files with a deliverable are checked (their effective
    content; a reviewed-unchanged file keeps its original), and only values
    whose enclosing profile governs this migration. A contradiction clears
    when an annotated change touching the value cites registry-recorded or
    plan-carried evidence for the departure (`known_evidence`).
    """
    if target_pricing is None:
        return []
    governed = [*source_reference_spellings(config), *target_reference_spellings(config)]
    couplings: dict[str, list[tuple[str, ...]]] = {}
    for finding in analysis.findings:
        metadata = finding.metadata or {}
        if metadata.get("value_kind") != "pricing" or not isinstance(metadata.get("config"), str):
            continue
        key_path = tuple(str(metadata.get("key_path", "")).split("."))
        couplings.setdefault(str(metadata["config"]), []).append(key_path)
    entries = {entry.source_path: entry for entry in log.entries if entry.kind == "file"}
    findings: list[ConsistencyFinding] = []
    for config_path, key_paths in sorted(couplings.items()):
        entry = entries.get(config_path)
        format = source_format(config_path)
        if entry is None or format is None:
            continue
        content = (
            _original_content(config, entry)
            if entry_is_unchanged(entry)
            else _effective_content(run_dir, entry)
        )
        original = _original_content(config, entry)
        if content is None:
            continue
        try:
            data = parse_structured_document(content, format)
            original_data = (
                parse_structured_document(original, format) if original is not None else data
            )
        except PromptDocumentError:
            continue
        for key_path in sorted(set(key_paths)):
            profile = _profile_model(original_data, key_path)
            if profile is not None and not any(
                spelling in profile or profile in spelling for spelling in governed
            ):
                continue  # another model's profile: not this migration's pricing
            check = classify_price(
                key_path,
                _lookup(data, key_path),
                target=target_pricing,
                source=source_pricing,
                as_of=as_of,
            )
            if check is None or check.verdict == "consistent":
                continue
            if check.verdict == "contradiction" and _departure_documented(
                entry, key_path[-1], known_evidence
            ):
                continue
            findings.append(
                ConsistencyFinding(
                    code=_PRICING_CODES[check.verdict],  # type: ignore[arg-type]
                    path=config_path,
                    message=check.message,
                )
            )
    return findings


def _departure_documented(entry: AdaptationEntry, key: str, known_evidence: set[str]) -> bool:
    """Whether a change touching `key` cites registry-recorded or plan-carried evidence."""
    for change in entry.annotated_changes:
        if key not in (change.adapted_anchor or "") and key not in (change.original_anchor or ""):
            continue
        if any(item.url and item.url in known_evidence for item in change.evidence):
            return True
    return False


class ConsistencyReport(StrictModel):
    """Rendered gate outcome carried on the finalization result."""

    schema_version: Literal["1"] = "1"
    findings: list[ConsistencyFinding] = Field(default_factory=list)

    @property
    def rendered(self) -> list[str]:
        return [finding.rendered for finding in self.findings]


def render_consistency_section(findings: list[ConsistencyFinding]) -> str:
    """The report section for the cross-surface consistency gate."""
    lines = [
        "## Cross-surface consistency",
        "",
        "Deterministic checks over the whole deliverable set (effective content, "
        "after review decisions): one selector-qualified target everywhere, no "
        "active source-model configuration remaining, no scanner-recognized "
        "coupling dropped without an explicit disposition, and no scanned "
        "application file still naming the source model without being accounted "
        "for. Coupling checks cover the kinds the scanner recognizes; the "
        "source-reference sweep covers every scanned file; adapted pricing values "
        "are compared unit-aware (per token / 1K / 1M) against the registry's "
        "target and source prices.",
        "",
    ]
    if findings:
        lines.extend(
            f"- **{finding.code}** `{finding.path}`: {finding.message}" for finding in findings
        )
    else:
        lines.append("- No cross-surface inconsistencies were found.")
    lines.append("")
    return "\n".join(lines)
