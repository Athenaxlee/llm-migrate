"""Maintainer evidence refresh (v1.6.1): keep recorded facts' freshness honest.

The registry's freshness windows are short by design (7 to 30 days), so
without a refresh mechanism every guided run found every topic stale within
weeks of the last review. This module is the maintainer's answer, and it is
deliberately narrow:

- it refetches ONLY the source URLs already recorded in canonical profiles —
  no discovery, no crawling, no new URLs;
- it hashes the page's visible text (scripts, styles, and markup stripped,
  whitespace collapsed) and compares against the baseline recorded in the
  model's `.registry-proposals/<model>/refresh-evidence.yaml`; page content
  is never stored;
- an unchanged page that still names the model lets it PROPOSE moving the
  `checked_at` date of the freshness categories that page supports; a page
  that no longer names the model (a lineup page that dropped a legacy model)
  vouches for nothing; a changed page is flagged for the maintainer to
  re-verify and never rebaselined silently; a URL without a baseline gets one
  recorded and proposes nothing;
- it writes proposals and baselines, never canonical registry files. The
  maintainer still promotes an approved proposal by hand, exactly like every
  other registry change (`research → proposal → review → canonical`).
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable
from datetime import date
from enum import StrEnum
from http.client import HTTPException
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError

import yaml
from pydantic import Field, ValidationError

from llm_migrate.adapters.evidence import SourceFetcher, fetch_https_source
from llm_migrate.core.models import FreshnessCategory, ModelProfile, SourceReference, StrictModel
from llm_migrate.core.registry import ModelRegistry, RegistryError
from llm_migrate.core.runstate import atomic_write_text

REFRESH_EVIDENCE_FILENAME = "refresh-evidence.yaml"

# What a recorded source's `supports` entries vouch for, in freshness terms.
_SUPPORT_CATEGORIES: dict[str, FreshnessCategory] = {
    "pricing": FreshnessCategory.PRICING,
    "lifecycle": FreshnessCategory.LIFECYCLE,
    "platforms": FreshnessCategory.AVAILABILITY,
    "capabilities": FreshnessCategory.CAPABILITIES,
    "parameters": FreshnessCategory.CAPABILITIES,
    "prompt_guidance": FreshnessCategory.PROMPTING_GUIDANCE,
}

_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


class RefreshError(RegistryError):
    """A refresh could not run (unknown model, unreadable baseline file)."""


class SourceBaseline(StrictModel):
    """The visible-text hash a recorded URL had when its facts were last verified."""

    url: str
    content_sha256: str
    content_length: int
    retrieved_at: date


class RefreshEvidenceLog(StrictModel):
    """Per-model baselines, kept beside the model's proposal bundle."""

    schema_version: Literal["1"] = "1"
    model: str
    baselines: list[SourceBaseline] = Field(default_factory=list)


class SourceRefreshStatus(StrEnum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    BASELINE_RECORDED = "baseline_recorded"
    # Fetched, but the visible text no longer names the model at all: such a
    # page cannot vouch for any of the model's facts, whatever its hash did.
    NO_MENTION = "no_mention"
    UNREACHABLE = "unreachable"
    REFUSED = "refused"


class SourceRefreshOutcome(StrictModel):
    url: str
    source_ids: list[str]
    supports: list[str]
    status: SourceRefreshStatus
    content_sha256: str | None = None
    content_length: int | None = None
    baseline_sha256: str | None = None
    error: str | None = None


class FreshnessRefreshProposal(StrictModel):
    category: FreshnessCategory
    current_checked_at: date | None
    proposed_checked_at: date
    supporting_urls: list[str]


class ModelRefreshReport(StrictModel):
    model: str
    sources: list[SourceRefreshOutcome]
    proposed_freshness: list[FreshnessRefreshProposal] = Field(default_factory=list)
    held_categories: list[str] = Field(default_factory=list)
    baselines_recorded: int = 0
    rebaselined: int = 0


class RefreshReport(StrictModel):
    schema_version: Literal["1"] = "1"
    refreshed_on: date
    models: list[ModelRefreshReport]
    message: str


def visible_text(payload: bytes) -> str:
    """The page's visible text: scripts, styles, and markup stripped, spaces collapsed."""
    text = payload.decode("utf-8", "replace")
    text = _SCRIPT_STYLE.sub(" ", text)
    text = html.unescape(_TAG.sub(" ", text))
    return _SPACE.sub(" ", text).strip()


def visible_text_digest(payload: bytes) -> tuple[str, int]:
    """sha256 and length of the page's visible text; markup noise never counts."""
    text = visible_text(payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)


def model_mentions(profile: ModelProfile) -> list[str]:
    """The spellings a page must contain (any one, case-insensitively) to name the model."""
    spellings = [profile.identity.canonical_name, profile.identity.display_name]
    spellings.extend(profile.identity.aliases)
    for platform in profile.platforms:
        spellings.append(platform.model_id)
        if platform.invocation is not None:
            spellings.extend(selector.model_id for selector in platform.invocation.selectors)
    return sorted({item.casefold() for item in spellings if item})


def recorded_sources(profile: ModelProfile) -> list[SourceReference]:
    """Every source recorded anywhere in the profile, top level or nested."""
    found: list[SourceReference] = []

    def walk(value: object) -> None:
        if isinstance(value, SourceReference):
            found.append(value)
        elif isinstance(value, StrictModel):
            for name in type(value).model_fields:
                walk(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)

    walk(profile)
    return found


def bundle_directory(proposals_root: Path, model: str) -> Path:
    """The model's proposal bundle: the directory whose proposal.yaml names it.

    Bundle directories are slugs of the proposal subject (for example
    `claude-sonnet-4-5` for `claude-sonnet-4-5-20250929`), so the canonical
    name is only the fallback.
    """
    if proposals_root.is_dir():
        for candidate in sorted(proposals_root.iterdir()):
            proposal = candidate / "proposal.yaml"
            if not proposal.is_file():
                continue
            try:
                raw = yaml.safe_load(proposal.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(raw, dict) and raw.get("model") == model:
                return candidate
    return proposals_root / model


def _baseline_path(proposals_root: Path, model: str) -> Path:
    return bundle_directory(proposals_root, model) / REFRESH_EVIDENCE_FILENAME


def bundle_holds(proposals_root: Path, model: str) -> dict[FreshnessCategory, str]:
    """Freshness categories the bundle's review put on hold, with the rationale.

    A review decision `decision: hold` whose field_path names
    `freshness.<category>` (several may be joined with `|`) records that the
    maintainer looked at the evidence and declined to move that date. The
    refresh honors it: no proposal for that category until the hold is lifted
    in the bundle.
    """
    proposal = bundle_directory(proposals_root, model) / "proposal.yaml"
    if not proposal.is_file():
        return {}
    try:
        raw = yaml.safe_load(proposal.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    review = raw.get("review")
    decisions = review.get("decisions") if isinstance(review, dict) else None
    holds: dict[FreshnessCategory, str] = {}
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, dict) or decision.get("decision") != "hold":
            continue
        for field_path in str(decision.get("field_path", "")).split("|"):
            prefix, _, name = field_path.strip().partition(".")
            if prefix != "freshness":
                continue
            try:
                category = FreshnessCategory(name)
            except ValueError:
                continue
            holds[category] = str(decision.get("rationale") or "held by the bundle review")
    return holds


def load_baselines(proposals_root: Path, model: str) -> RefreshEvidenceLog:
    path = _baseline_path(proposals_root, model)
    if not path.is_file():
        return RefreshEvidenceLog(model=model)
    try:
        return RefreshEvidenceLog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise RefreshError(f"unreadable refresh baselines {path}: {exc}") from exc


def _write_baselines(proposals_root: Path, log: RefreshEvidenceLog) -> Path:
    path = _baseline_path(proposals_root, log.model)
    if not path.parent.is_dir():
        raise RefreshError(
            f"no proposal bundle directory for {log.model!r} under {proposals_root}; "
            "baselines live beside the model's reviewed bundle"
        )
    atomic_write_text(path, yaml.safe_dump(log.model_dump(mode="json"), sort_keys=False))
    return path


def _group_by_url(sources: Iterable[SourceReference]) -> dict[str, tuple[list[str], list[str]]]:
    grouped: dict[str, tuple[list[str], list[str]]] = {}
    for source in sources:
        if source.url is None:
            continue
        ids, supports = grouped.setdefault(str(source.url), ([], []))
        if source.id not in ids:
            ids.append(source.id)
        for item in source.supports:
            if item not in supports:
                supports.append(item)
    return dict(sorted(grouped.items()))


def refresh_model_evidence(
    profile: ModelProfile,
    baselines: RefreshEvidenceLog,
    *,
    fetcher: SourceFetcher,
    timeout: float,
    today: date,
    rebaseline: bool,
    holds: dict[FreshnessCategory, str] | None = None,
) -> tuple[ModelRefreshReport, RefreshEvidenceLog]:
    """Refetch one profile's recorded URLs; return the report and updated baselines."""
    model = profile.identity.canonical_name
    held_by_review = holds or {}
    mentions = model_mentions(profile)
    known = {item.url: item for item in baselines.baselines}
    outcomes: list[SourceRefreshOutcome] = []
    updated: dict[str, SourceBaseline] = dict(known)
    recorded = rebaselined = 0
    for url, (ids, supports) in _group_by_url(recorded_sources(profile)).items():
        baseline = known.get(url)
        try:
            payload = fetcher(url, timeout)
        except (ValueError, HTTPError, URLError, HTTPException, TimeoutError, OSError) as exc:
            failure = (
                SourceRefreshStatus.REFUSED
                if isinstance(exc, ValueError)
                else SourceRefreshStatus.UNREACHABLE
            )
            outcomes.append(
                SourceRefreshOutcome(
                    url=url,
                    source_ids=ids,
                    supports=supports,
                    status=failure,
                    baseline_sha256=baseline.content_sha256 if baseline else None,
                    error=str(exc),
                )
            )
            continue
        text = visible_text(payload)
        digest, length = hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)
        named = any(item in text.casefold() for item in mentions)
        if baseline is None:
            status = SourceRefreshStatus.BASELINE_RECORDED
            updated[url] = SourceBaseline(
                url=url, content_sha256=digest, content_length=length, retrieved_at=today
            )
            recorded += 1
        elif baseline.content_sha256 == digest:
            status = SourceRefreshStatus.UNCHANGED if named else SourceRefreshStatus.NO_MENTION
        else:
            status = SourceRefreshStatus.CHANGED
            if rebaseline:
                updated[url] = SourceBaseline(
                    url=url, content_sha256=digest, content_length=length, retrieved_at=today
                )
                rebaselined += 1
        outcomes.append(
            SourceRefreshOutcome(
                url=url,
                source_ids=ids,
                supports=supports,
                status=status,
                content_sha256=digest,
                content_length=length,
                baseline_sha256=baseline.content_sha256 if baseline else None,
                error=None if named else "the page's visible text no longer names this model",
            )
        )
    proposals: list[FreshnessRefreshProposal] = []
    held: list[str] = []
    for category in FreshnessCategory:
        supporting = [
            outcome
            for outcome in outcomes
            if category in {_SUPPORT_CATEGORIES.get(item) for item in outcome.supports}
        ]
        by_status: dict[SourceRefreshStatus, list[str]] = {}
        for outcome in supporting:
            by_status.setdefault(outcome.status, []).append(outcome.url)
        unchanged = by_status.get(SourceRefreshStatus.UNCHANGED, [])
        changed = by_status.get(SourceRefreshStatus.CHANGED, [])
        metadata = profile.freshness.get(category)
        if category in held_by_review:
            held.append(
                f"{category.value}: on hold by the bundle's review decision — "
                f"{held_by_review[category]}"
            )
        elif unchanged and not changed:
            proposals.append(
                FreshnessRefreshProposal(
                    category=category,
                    current_checked_at=metadata.checked_at if metadata else None,
                    proposed_checked_at=today,
                    supporting_urls=unchanged,
                )
            )
        elif changed:
            held.append(
                f"{category.value}: re-verify the changed page(s) {', '.join(changed)} before "
                "moving checked_at (use --rebaseline once verified)"
            )
        elif not supporting:
            held.append(f"{category.value}: no recorded source supports it; nothing to refresh")
        elif SourceRefreshStatus.BASELINE_RECORDED in by_status:
            held.append(
                f"{category.value}: first refresh — baselines recorded for "
                f"{', '.join(by_status[SourceRefreshStatus.BASELINE_RECORDED])}; verify the "
                "facts by hand this time"
            )
        elif SourceRefreshStatus.NO_MENTION in by_status:
            held.append(
                f"{category.value}: the supporting page(s) "
                f"{', '.join(by_status[SourceRefreshStatus.NO_MENTION])} no longer name this "
                "model; record a source that does"
            )
        else:
            held.append(
                f"{category.value}: every supporting page was unreachable or refused "
                f"({', '.join(url for urls in by_status.values() for url in urls)}); retry later"
            )
    report = ModelRefreshReport(
        model=model,
        sources=outcomes,
        proposed_freshness=proposals,
        held_categories=held,
        baselines_recorded=recorded,
        rebaselined=rebaselined,
    )
    log = RefreshEvidenceLog(model=model, baselines=[updated[url] for url in sorted(updated)])
    return report, log


def refresh_evidence(
    registry: ModelRegistry,
    proposals_root: Path,
    *,
    models: list[str] | None = None,
    fetcher: SourceFetcher | None = None,
    timeout: float = 5.0,
    today: date | None = None,
    rebaseline: bool = False,
    record_baselines: bool = True,
) -> RefreshReport:
    """Refresh recorded evidence for the named (default: every non-fixture) models.

    Writes only `<proposals_root>/<model>/refresh-evidence.yaml` baselines
    (when `record_baselines`), never a registry file.
    """
    effective_today = today or date.today()
    fetch = fetcher or fetch_https_source
    profiles = [profile for profile in registry.all() if not profile.fixture]
    if models:
        by_name = {profile.identity.canonical_name: profile for profile in registry.all()}
        missing = sorted(set(models) - set(by_name))
        if missing:
            raise RefreshError(f"unknown canonical model(s): {', '.join(missing)}")
        profiles = [by_name[name] for name in models]
    reports: list[ModelRefreshReport] = []
    for profile in profiles:
        name = profile.identity.canonical_name
        report, log = refresh_model_evidence(
            profile,
            load_baselines(proposals_root, name),
            fetcher=fetch,
            timeout=timeout,
            today=effective_today,
            rebaseline=rebaseline,
            holds=bundle_holds(proposals_root, name),
        )
        if record_baselines and (report.baselines_recorded or report.rebaselined):
            _write_baselines(proposals_root, log)
        reports.append(report)
    proposed = sum(len(report.proposed_freshness) for report in reports)
    changed = sum(
        1
        for report in reports
        for outcome in report.sources
        if outcome.status is SourceRefreshStatus.CHANGED
    )
    recorded = sum(report.baselines_recorded for report in reports)
    message = (
        f"{len(reports)} model(s) refreshed: {proposed} freshness update(s) proposed, "
        f"{changed} changed page(s) need maintainer re-verification, {recorded} baseline(s) "
        + ("recorded" if record_baselines else "computed but NOT written (report only)")
        + ". Canonical registry files were not modified; promote approved updates "
        "through the model's proposal bundle."
    )
    return RefreshReport(refreshed_on=effective_today, models=reports, message=message)


def refresh_report_markdown(report: RefreshReport) -> str:
    lines = [f"# Evidence refresh {report.refreshed_on.isoformat()}", "", report.message, ""]
    for model in report.models:
        lines.extend((f"## {model.model}", ""))
        for outcome in model.sources:
            detail = f" ({outcome.error})" if outcome.error else ""
            supports = ", ".join(outcome.supports)
            lines.append(f"- {outcome.status.value}: {outcome.url} [{supports}]{detail}")
        if model.proposed_freshness:
            lines.extend(("", "Proposed `freshness` updates:", ""))
            for item in model.proposed_freshness:
                lines.append(
                    f"- {item.category.value}: checked_at {item.current_checked_at} → "
                    f"{item.proposed_checked_at} (unchanged: {', '.join(item.supporting_urls)})"
                )
        if model.held_categories:
            lines.extend(("", "Held:", "", *[f"- {item}" for item in model.held_categories]))
        lines.append("")
    return "\n".join(lines)


def write_refresh_report(report: RefreshReport, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"refresh-{report.refreshed_on.isoformat()}"
    atomic_write_text(
        output_directory / f"{stem}.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False),
    )
    atomic_write_text(output_directory / f"{stem}.md", refresh_report_markdown(report))
    return output_directory / f"{stem}.md"
