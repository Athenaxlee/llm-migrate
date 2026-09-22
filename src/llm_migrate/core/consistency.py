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
from pathlib import Path
from typing import Literal

from pydantic import Field

from llm_migrate.core.invocation_identity import references_bare_alone
from llm_migrate.core.models import ApplicationAnalysis, CouplingKind, StrictModel
from llm_migrate.core.workspace import (
    AdaptationEntry,
    AdaptationLog,
    MigrationRunConfig,
    effective_target_id,
    entry_is_unchanged,
    source_detection_spellings,
    target_reference_spellings,
    target_spellings,
)

ConsistencyCode = Literal[
    "source_reference_remains",
    "bare_target_reference",
    "mixed_target_selectors",
    "target_reference_missing",
    "dropped_coupling_undisposed",
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


def check_cross_surface_consistency(
    run_dir: Path,
    config: MigrationRunConfig,
    analysis: ApplicationAnalysis,
    log: AdaptationLog,
) -> list[ConsistencyFinding]:
    """Deterministic checks over ALL deliverables and unchanged claims together."""
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
        remaining = [item for item in source_forms if item in effective]
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
    return findings


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
        "active source-model configuration remaining, and no scanner-recognized "
        "coupling dropped without an explicit disposition. The gate covers the "
        "coupling kinds the scanner recognizes, no more.",
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
