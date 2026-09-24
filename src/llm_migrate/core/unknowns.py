"""Typed, actionable migration unknowns (V1.6.0-b).

Every emitter here returns `MigrationUnknown` records whose four direction
fields are populated from THIS application's scan: what the unknown is about,
why it matters here, the exact next action (a ready-to-run tool call or CLI
command), and what closes it. Actions address the run through
`RUN_DIR_PLACEHOLDER`; a guided run substitutes its directory
(`with_run_dir`), so a run's actions are copy-paste exact.

Unknowns the scan already answers are emitted `closed_by_scan`, and ones a
recorded user decision or observation answered are `closed_by_action`, each
with its reason — the report can always say how an unknown was closed.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm_migrate.core.models import (
    RUN_DIR_PLACEHOLDER,
    ApplicationAnalysis,
    CompatibilityAssessment,
    CouplingKind,
    InvocationMigrationSpec,
    MigrationPlan,
    MigrationUnknown,
    ModelDifference,
    PromptSource,
    ResolvedModel,
    StrictModel,
    UnknownKind,
    unknown_id,
)
from llm_migrate.core.probes import contested_setting, probe_relative_path, probe_supported

OBSERVATION_CLOSING = (
    "Closed when the target's actual behavior is recorded with record_observation "
    "(after the generated contract test, a BYOK evaluation, or a probe), or when "
    "reviewed research resolves the registry fact."
)


def tool_call(tool: str, **arguments: Any) -> str:
    """A syntactically valid Python-call rendering of one tool invocation."""
    rendered = ", ".join(f"{name}={json.dumps(value)}" for name, value in arguments.items())
    return f"{tool}({rendered})"


def _unknown(
    kind: UnknownKind,
    discriminator: str,
    *,
    subject: str,
    message: str,
    why: str,
    action: str | None = None,
    closing: str,
    evidence_urls: Sequence[str] = (),
    data: dict[str, Any] | None = None,
    status: str = "open",
    closed_reason: str | None = None,
) -> MigrationUnknown:
    identifier = unknown_id(kind, discriminator)
    return MigrationUnknown(
        id=identifier,
        kind=kind,
        subject=subject,
        why_it_matters=why,
        action=action
        or tool_call(
            "record_observation",
            run_dir=RUN_DIR_PLACEHOLDER,
            subject=identifier,
            outcome="<what the target actually did>",
            evidence="<test output, request id, or source URL>",
        ),
        closing_condition=closing,
        message=message,
        status=status,  # type: ignore[arg-type]
        closed_reason=closed_reason,
        evidence_urls=list(evidence_urls),
        data=data,
    )


# -- prompt discovery ---------------------------------------------------------

CANDIDATE_MARKER = " contains prompt-like keys but nothing references it"


def candidate_unknown(source: PromptSource) -> MigrationUnknown:
    """An unreferenced candidate prompt file with no selected sibling."""
    keys = ", ".join(item.key or "(whole file)" for item in source.components)
    return _unknown(
        UnknownKind.PROMPT_CANDIDATE,
        source.path,
        subject=f"Unreferenced candidate prompt file {source.path}",
        message=(
            f"{source.path}{CANDIDATE_MARKER}; it was not prepared automatically. "
            "Include it with add_prompt_sources (or prompt_sources at start) if it is live."
        ),
        why=(
            f"{source.path} holds prompt components ({keys}) that no scanned code or "
            "configuration references; if the application loads it at runtime, those "
            "prompts are not being migrated or validated."
        ),
        action=tool_call("add_prompt_sources", run_dir=RUN_DIR_PLACEHOLDER, paths=[source.path]),
        closing=(
            f"Closed when {source.path} is added as a prompt source, or dismissed as not a "
            "live prompt with add_prompt_sources(run_dir, paths=[], "
            f'dismiss=["{source.path}"], rationale="<the user\'s reason>").'
        ),
        data={"path": source.path},
    )


@dataclass(frozen=True)
class ConsumerRef:
    """One prompt consumer with the literal keys it reads and matching sources."""

    location: str
    keyword: str
    resolution: str
    access_keys: list[str] = field(default_factory=list)
    matching_sources: list[str] = field(default_factory=list)
    confirmed_source: str | None = None


def consumer_refs(
    application: ApplicationAnalysis, excluded_paths: set[str] | frozenset[str] = frozenset()
) -> list[ConsumerRef]:
    """Every dynamic, confirmed, or dismissed prompt consumer with its key matches.

    `matching_sources` names the in-scope prompt sources whose component keys
    contain every literal key the consumer reads.
    """
    component_keys = {
        source.path: {
            (component.key or "").split("[", 1)[0]
            for component in source.components
            if component.key
        }
        for source in application.prompt_sources
        if source.path not in excluded_paths
    }
    refs: list[ConsumerRef] = []
    for finding in application.findings:
        metadata = finding.metadata or {}
        if finding.kind is not CouplingKind.PROMPT or not finding.detail.startswith(
            "supplies prompt content"
        ):
            continue
        resolution = str(metadata.get("resolution"))
        confirmed = bool(metadata.get("confirmed"))
        if resolution not in {"dynamic", "dismissed"} and not confirmed:
            continue
        access = [str(key) for key in metadata.get("access_keys") or []]
        sources = [str(item) for item in metadata.get("sources") or []]
        refs.append(
            ConsumerRef(
                location=f"{finding.location.path}:{finding.location.line}:{finding.value}",
                keyword=str(finding.value),
                resolution="confirmed" if confirmed else resolution,
                access_keys=access,
                matching_sources=sorted(
                    path for path, keys in component_keys.items() if access and set(access) <= keys
                ),
                confirmed_source=sources[0] if confirmed and sources else None,
            )
        )
    return refs


def consumer_unknown(ref: ConsumerRef) -> MigrationUnknown:
    """One dynamic prompt consumer (open), or its recorded confirmation/dismissal."""
    reads = (
        f", reading {', '.join(repr(key) for key in ref.access_keys)}" if ref.access_keys else ""
    )
    if len(ref.matching_sources) == 1:
        source = ref.matching_sources[0]
        hint = f" Only {source} holds those keys."
    elif ref.matching_sources:
        source = f"<ask the user: one of {', '.join(ref.matching_sources)}>"
        hint = f" Files holding those keys: {', '.join(ref.matching_sources)}."
    else:
        source = "<the prompt file it reads>"
        hint = ""
    status, reason = "open", None
    if ref.resolution == "confirmed":
        status, reason = "closed_by_action", f"confirmed by the user to read {ref.confirmed_source}"
    elif ref.resolution == "dismissed":
        status, reason = "closed_by_action", "dismissed by the user as runtime-built content"
    return _unknown(
        UnknownKind.DYNAMIC_PROMPT_CONSUMER,
        ref.location,
        subject=f"Dynamic prompt consumer {ref.location} ({ref.keyword})",
        message=(
            f"Prompt consumer {ref.location} supplies dynamically built {ref.keyword} "
            "content with no statically resolvable prompt source."
        ),
        why=(
            f"The {ref.keyword} content passed to the model at {ref.location} is built at "
            f"runtime{reads}; with no resolved prompt source its text is neither migrated "
            f"nor validated for the target.{hint}"
        ),
        action=tool_call(
            "confirm_prompt_consumer",
            run_dir=RUN_DIR_PLACEHOLDER,
            location=ref.location,
            source_path=source,
        ),
        closing=(
            "Closed when the consumer is confirmed to read a prompt file, or dismissed as "
            "genuinely runtime-built content with add_prompt_sources(run_dir, paths=[], "
            f'dismiss=["{ref.location}"], rationale="<the user\'s reason>").'
        ),
        data={
            "location": ref.location,
            "keyword": ref.keyword,
            "access_keys": ref.access_keys,
            "matching_sources": ref.matching_sources,
        },
        status=status,
        closed_reason=reason,
    )


# -- compatibility and model differences ------------------------------------


def compatibility_unknown(assessment: CompatibilityAssessment) -> MigrationUnknown:
    files = sorted({item.path for item in assessment.locations})
    where = f" at {', '.join(files[:3])}" if files else ""
    return _unknown(
        UnknownKind.COMPATIBILITY,
        assessment.concern,
        subject=f"Target {assessment.concern} compatibility",
        message=assessment.rationale,
        why=(
            f"The application's {assessment.concern} usage{where} has no statically "
            f"established target mapping: {assessment.rationale}"
        ),
        closing=OBSERVATION_CLOSING,
        data={"concern": assessment.concern, "files": files},
    )


def _value_text(value: Any) -> str:
    if value is None:
        return "no recorded value"
    if isinstance(value, Enum):
        return repr(value.value)
    return repr(value)


def difference_unknown(difference: ModelDifference, *, used: bool) -> MigrationUnknown:
    """An unknown model difference; `used=False` means the scan answered it."""
    field_name = difference.field or difference.category
    files = sorted({item.path for item in difference.locations})
    usage = (
        f" The application relies on it at {', '.join(files[:3])}."
        if files
        else " The target's value applies to every call the application makes."
    )
    evidence = list(
        dict.fromkeys(
            url for url in (difference.source_evidence_url, difference.target_evidence_url) if url
        )
    )
    return _unknown(
        UnknownKind.MODEL_DIFFERENCE,
        f"{difference.category}|{field_name}",
        subject=f"Target {field_name} ({difference.category})",
        message=difference.migration_impact,
        why=(
            f"The reviewed registry records {field_name} as "
            f"{_value_text(difference.source_value)} on the source and "
            f"{_value_text(difference.target_value)} on the target, so compatibility must "
            f"not be assumed.{usage}"
        ),
        closing=OBSERVATION_CLOSING,
        evidence_urls=evidence,
        data={"category": difference.category, "field": field_name},
        status="open" if used else "closed_by_scan",
        closed_reason=None if used else f"the scan shows the application does not use {field_name}",
    )


# -- contested registry evidence --------------------------------------------


def _field_parameter(field_path: str) -> str | None:
    parts = field_path.split(".")
    return parts[parts.index("parameters") + 1] if "parameters" in parts[:-1] else None


def contested_unknowns(
    target: ResolvedModel,
    platform: str,
    application: ApplicationAnalysis,
    invocation: InvocationMigrationSpec | None,
) -> list[MigrationUnknown]:
    """Registry evidence conflicts on the target that govern this platform.

    A conflict scoped to another platform (`platforms.<other>.…`) does not
    govern this migration. Each emitted unknown carries both statements and
    their source URLs; when the conflict is empirically testable on the
    target call, the action is the emitted BYOK probe script.
    """
    urls = {source.id: str(source.url) for source in target.profile.sources if source.url}
    detected = application.requirements.required_parameters
    unknowns: list[MigrationUnknown] = []
    for conflict in target.profile.evidence_conflicts:
        path = conflict.field_path
        if path.startswith("platforms.") and not path.startswith(f"platforms.{platform}."):
            continue
        parameter = _field_parameter(path)
        if parameter is not None and (
            parameter in detected or parameter.split("_")[-1] in detected
        ):
            usage = f"The application sets {parameter!r}, so its behavior depends on this fact."
        else:
            usage = (
                "The target's default governs every call the application makes, and any "
                "adaptation that changes this setting depends on the contested fact."
            )
        testable = probe_supported(invocation, path)
        probe_path = probe_relative_path(path) if testable else None
        statements = list(conflict.statements)
        unknowns.append(
            _unknown(
                UnknownKind.CONTESTED_EVIDENCE,
                path,
                subject=f"Contested registry fact {path}",
                message=(f"Registry evidence for {path} is conflicting: " + " / ".join(statements)),
                why=" ".join(
                    [
                        "Reviewed sources disagree: " + " / ".join(statements) + ".",
                        usage,
                        *([conflict.notes] if conflict.notes else []),
                    ]
                ),
                action=(f"python {RUN_DIR_PLACEHOLDER}/{probe_path}" if probe_path else None),
                closing=(
                    "Closed when the probe's printed result (or another empirical check) is "
                    "recorded with record_observation for this unknown's id; promoting it to "
                    "the registry is a separate propose_registry_update."
                ),
                evidence_urls=[urls[item] for item in conflict.source_ids if item in urls],
                data={
                    "field_path": path,
                    "statements": statements,
                    "probe_path": probe_path,
                },
            )
        )
    return unknowns


# -- run-level application ----------------------------------------------------


def _substitute_run_dir(action: str, run_dir: str) -> str:
    """Fill the run directory per action shape: shell-quoted for a command,
    JSON-escaped inside a Python-literal tool call."""
    if action.startswith("python "):
        return "python " + shlex.quote(
            action[len("python ") :].replace(RUN_DIR_PLACEHOLDER, run_dir)
        )
    return action.replace(RUN_DIR_PLACEHOLDER, json.dumps(run_dir)[1:-1])


def with_run_dir(unknowns: Sequence[MigrationUnknown], run_dir: str) -> list[MigrationUnknown]:
    """Substitute the run directory into every action."""
    return [
        item.model_copy(update={"action": _substitute_run_dir(item.action, run_dir)})
        if RUN_DIR_PLACEHOLDER in item.action
        else item
        for item in unknowns
    ]


def close_by_action(
    unknowns: Sequence[MigrationUnknown], reasons: Mapping[str, str]
) -> list[MigrationUnknown]:
    """Mark unknowns (by id) closed by a recorded user action, with the reason."""
    return [
        item.model_copy(update={"status": "closed_by_action", "closed_reason": reasons[item.id]})
        if item.id in reasons and item.status != "closed_by_scan"
        else item
        for item in unknowns
    ]


def sort_unknowns(unknowns: Sequence[MigrationUnknown]) -> list[MigrationUnknown]:
    order = {kind: index for index, kind in enumerate(UnknownKind)}
    unique = {item.id: item for item in unknowns}
    return sorted(unique.values(), key=lambda item: (order[item.kind], item.subject, item.id))


# -- contested facts in change review (V1.6.0-c) ------------------------------


class ContestedFact(StrictModel):
    """One contested registry fact a change may depend on, for review marks."""

    unknown_id: str
    field_path: str
    statements: list[str]
    evidence_urls: list[str]
    parameter: str | None = None
    setting_tokens: list[str]
    open: bool
    action: str
    closed_reason: str | None = None


def _setting_tokens(setting: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(setting, dict):
        for key, value in setting.items():
            tokens.append(str(key))
            tokens.extend(_setting_tokens(value))
    elif isinstance(setting, str):
        tokens.append(setting)
    return tokens


def contested_facts(plan: MigrationPlan) -> list[ContestedFact]:
    """Every contested-evidence unknown of a plan as a review-mark matcher."""
    facts: list[ContestedFact] = []
    for unknown in plan.unknowns:
        if unknown.kind is not UnknownKind.CONTESTED_EVIDENCE:
            continue
        path = str((unknown.data or {}).get("field_path", ""))
        facts.append(
            ContestedFact(
                unknown_id=unknown.id,
                field_path=path,
                statements=[str(item) for item in (unknown.data or {}).get("statements", [])],
                evidence_urls=list(unknown.evidence_urls),
                parameter=_field_parameter(path),
                setting_tokens=_setting_tokens(contested_setting(path)),
                open=unknown.is_open,
                action=unknown.action,
                closed_reason=unknown.closed_reason,
            )
        )
    return facts


def contested_marks(change: Any, facts: Sequence[ContestedFact]) -> list[str]:
    """CONTESTED marks for one annotated change that depends on an open conflict.

    A change depends on a contested fact when its adapted text exercises the
    contested setting (every key and value of the probe setting appears, for
    example `thinking` and `disabled`), or when its why/evidence names the
    contested field path or parameter. Citing a conflict's source URL alone
    is not enough: those documents back many uncontested facts. A recorded
    observation closing the fact clears the mark.
    """
    marks: list[str] = []
    adapted = str(getattr(change, "adapted_anchor", "") or "")
    cited = " ".join(
        [str(getattr(change, "why", ""))]
        + [str(getattr(item, "reference", "")) for item in getattr(change, "evidence", [])]
    )
    for fact in facts:
        if not fact.open:
            continue
        # Carrying a contested setting over from the source call still makes
        # the adapted call depend on the fact, so the original anchor is not
        # consulted.
        exercises = bool(fact.setting_tokens) and all(
            token in adapted for token in fact.setting_tokens
        )
        names = fact.field_path in cited or bool(fact.parameter and fact.parameter in cited)
        if not (exercises or names):
            continue
        marks.append(
            f"CONTESTED: depends on {fact.field_path} [{fact.unknown_id}], where reviewed "
            f"sources disagree — {' / '.join(fact.statements)} (sources: "
            f"{', '.join(fact.evidence_urls) or 'none recorded'}). Resolve empirically before "
            f"accepting: {fact.action}, then record_observation for {fact.unknown_id}."
        )
    return marks
