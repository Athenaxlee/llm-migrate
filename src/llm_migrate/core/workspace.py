"""Guided migration run workspace: layout, adaptation deliverables, reporting.

A run workspace holds every artifact one migration produces, by default beneath
`.llm-migrate/runs/<run-id>/` in the analyzed application. The workspace never
writes into the application source tree: adapted prompts and adapted source
files are finalized review candidates beneath the run's `output/` directory,
and applying them to the application remains an explicit human step.
"""

from __future__ import annotations

import ast
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError

from llm_migrate.analyzers.prompt import structural_sections
from llm_migrate.core.agent_research import RUN_ID_PATTERN, ModelEndpointIdentity
from llm_migrate.core.annotations import (
    AnnotatedChange,
    annotation_problems,
    decoded_view,
    standalone_annotation_problems,
    with_change_ids,
)
from llm_migrate.core.invocation_identity import (
    qualify_bare_references,
    references_bare_alone,
)
from llm_migrate.core.models import (
    ComparisonSeverity,
    MigrationAdvice,
    MigrationPlan,
    ModelMatchResult,
    PromptMigrationSpec,
    PromptSourceFormat,
    PromptValidationResult,
    StrictModel,
    ValidationLevel,
)
from llm_migrate.core.prompt_documents import (
    STRUCTURED_SUFFIXES,
    build_candidate_document,
    document_mentions,
    evaluate_prompt_submission,
    is_prompt_bearing,
    source_format,
)
from llm_migrate.core.runstate import atomic_write_text, run_state_lock

RUN_CONFIG_FILENAME = "migration.yaml"
CHANGES_FILENAME = "changes.yaml"
CHANGE_DECISIONS_FILENAME = "change-decisions.yaml"
DEFAULT_RUNS_SUBDIR = Path(".llm-migrate") / "runs"


class WorkspaceError(ValueError):
    """A run workspace operation cannot proceed as requested."""


class MigrationRunConfig(StrictModel):
    """Durable identity of one migration run, persisted as `migration.yaml`.

    `*_model_id` is the bare platform model id (canonical identity);
    `*_invocation_model_id` is the id the platform accepts for an on-demand
    call (selector-qualified when a selector was chosen). The invocation
    fields are additive: schema version stays "1" and configs written before
    v1.5.0-a still load, behaving exactly as before.
    """

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1, pattern=RUN_ID_PATTERN)
    application_root: str
    source: ModelEndpointIdentity
    target: ModelEndpointIdentity
    source_model_id: str
    target_model_id: str
    source_invocation_model_id: str | None = None
    target_invocation_model_id: str | None = None
    source_invocation_selector: str | None = None
    target_invocation_selector: str | None = None
    target_invocation_requires_selector: bool = False
    source_model_spellings: list[str] = Field(default_factory=list)
    target_model_spellings: list[str] = Field(default_factory=list)
    created_on: date
    prompt_sources: list[str] = Field(default_factory=list)


def source_spellings(config: MigrationRunConfig) -> list[str]:
    """Every reviewed spelling that names the source model on its platform."""
    return config.source_model_spellings or [config.source_model_id]


def target_spellings(config: MigrationRunConfig) -> list[str]:
    """Every reviewed spelling that names the target model on its platform."""
    return config.target_model_spellings or [config.target_model_id]


def effective_target_id(config: MigrationRunConfig) -> str:
    """The id the application must reference to invoke the target on demand."""
    return config.target_invocation_model_id or config.target_model_id


def effective_source_id(config: MigrationRunConfig) -> str:
    """The id the application referenced to invoke the source on demand."""
    return config.source_invocation_model_id or config.source_model_id


class MigrationRunPaths(StrictModel):
    """Where one run keeps its artifacts; all deliverables live under output/."""

    run_dir: str
    config_path: str
    request_path: str
    decisions_path: str
    research_dir: str
    review_dir: str
    output_dir: str
    prompts_dir: str
    files_dir: str
    manifest_path: str
    report_path: str
    changes_path: str
    change_decisions_path: str


class ResearchNeed(StrictModel):
    """Whether and why bounded research would improve this run."""

    level: Literal["none", "recommended"]
    reasons: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    request_path: str | None = None


class MigrationRunStart(StrictModel):
    """Result of starting (or attempting to start) a guided migration run."""

    schema_version: Literal["1"] = "1"
    status: Literal["ready", "needs_confirmation"]
    source_match: ModelMatchResult
    target_match: ModelMatchResult
    run: MigrationRunConfig | None = None
    paths: MigrationRunPaths | None = None
    research: ResearchNeed | None = None
    warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class GuidanceItem(StrictModel):
    """One evidence-linked guidance line with a stable content-derived id.

    The id is deterministic over the text (like blocker ids), so the same
    guidance keeps the same id across worklist regenerations and a recorded
    disposition keeps matching it; changed guidance gets a new id and must be
    disposed again.
    """

    id: str
    text: str


def guidance_item(text: str) -> GuidanceItem:
    """Build a guidance item with its deterministic stable id."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return GuidanceItem(id=f"g:{digest}", text=text)


class GuidanceDisposition(StrictModel):
    """The submitting agent's explicit verdict on one guidance item.

    Every prompt submission must dispose every guidance item of its task
    (own guidance plus the worklist's shared prompt guidance): `applied`
    (the adaptation acts on it), `not_applicable` (it does not apply to this
    prompt), or `declined` (deliberately not applied — requires a note).
    `guidance` is the resolved text, filled in at acceptance so the recorded
    log is self-contained.
    """

    guidance_id: str
    disposition: Literal["applied", "not_applicable", "declined"]
    note: str = ""
    guidance: str = ""


class AdaptationEntry(StrictModel):
    """One submitted, validated adaptation deliverable."""

    kind: Literal["prompt", "file"]
    source_path: str
    output_path: str
    rationale: str
    changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_sha256: str | None = None
    adapted_sha256: str
    submitted_on: date
    unchanged: bool = False
    guidance_dispositions: list[GuidanceDisposition] = Field(default_factory=list)
    annotated_changes: list[AnnotatedChange] = Field(default_factory=list)


def entry_is_unchanged(entry: AdaptationEntry) -> bool:
    """Whether a deliverable is a reviewed no-change copy of its original.

    Entries recorded before the `unchanged` field existed carry only the
    fingerprints, so equal hashes count as unchanged too.
    """
    return entry.unchanged or (
        entry.source_sha256 is not None and entry.source_sha256 == entry.adapted_sha256
    )


class AdaptationLog(StrictModel):
    """Reviewable record of every adaptation deliverable in one run.

    Schema version 2 entries carry hunk-anchored `annotated_changes`; version
    1 logs (free-text changes only) still load, and every save writes the
    current version.
    """

    schema_version: Literal["1", "2"] = "2"
    run_id: str
    entries: list[AdaptationEntry] = Field(default_factory=list)


class ChangeDecision(StrictModel):
    """One durable user decision on one annotated change.

    The decision is keyed to the submission's content fingerprints, so a
    resubmission of the file makes it stale (reported, never silently
    applied) — the same contract blocker decisions follow.
    """

    source_path: str
    kind: Literal["prompt", "file"]
    change_id: str
    decision: Literal["accepted", "rejected"]
    note: str = ""
    decided_on: date
    source_sha256: str | None = None
    adapted_sha256: str


class ChangeDecisionLog(StrictModel):
    """Durable record of every change review decision (change-decisions.yaml)."""

    schema_version: Literal["1"] = "1"
    run_id: str
    decisions: list[ChangeDecision] = Field(default_factory=list)


def load_change_decision_log(run_dir: Path, run_id: str) -> ChangeDecisionLog:
    """The run's change decision log; a log from another run is refused."""
    path = Path(run_dir) / CHANGE_DECISIONS_FILENAME
    if not path.is_file():
        return ChangeDecisionLog(run_id=run_id)
    try:
        log = ChangeDecisionLog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise WorkspaceError(f"invalid change decision log {path}: {exc}") from exc
    if log.run_id != run_id:
        raise WorkspaceError(
            f"{path} belongs to run {log.run_id!r}, not {run_id!r}; change decisions "
            "are never carried between runs"
        )
    return log


def save_change_decision_log(run_dir: Path, log: ChangeDecisionLog) -> Path:
    path = Path(run_dir) / CHANGE_DECISIONS_FILENAME
    atomic_write_text(path, yaml.safe_dump(log.model_dump(mode="json"), sort_keys=False))
    return path


def upsert_change_decision(log: ChangeDecisionLog, decision: ChangeDecision) -> ChangeDecisionLog:
    """One decision per (source_path, change_id); a new one replaces the old."""
    kept = [
        item
        for item in log.decisions
        if not (item.source_path == decision.source_path and item.change_id == decision.change_id)
    ]
    return log.model_copy(update={"decisions": [*kept, decision]})


def decision_matches_entry(decision: ChangeDecision, entry: AdaptationEntry) -> bool:
    """Whether a recorded decision still applies to the current submission."""
    return (
        decision.source_path == entry.source_path
        and decision.kind == entry.kind
        and decision.source_sha256 == entry.source_sha256
        and decision.adapted_sha256 == entry.adapted_sha256
        and any(change.id == decision.change_id for change in entry.annotated_changes)
    )


class PromptAdaptationTask(StrictModel):
    """One prompt file the host agent must rewrite for the target model.

    `verbatim_source` is the ORIGINAL prompt content, unmodified (for
    structured documents, the document rebuilt from its verbatim components).
    It is the starting point for the agent's own adaptation, never a proposed
    adaptation, and must never be presented to the user as one.
    """

    source_path: str
    output_path: str
    status: Literal["pending", "submitted"]
    verbatim_source: str
    components: list[str] = Field(default_factory=list)
    guidance: list[GuidanceItem] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class FileAdaptationTask(StrictModel):
    """One application file the host agent must adapt for the target model.

    Each required change carries a stable id; file submissions dispose them
    through `guidance_dispositions` exactly like prompt guidance.
    """

    source_path: str
    output_path: str
    status: Literal["pending", "submitted"]
    required_changes: list[GuidanceItem] = Field(default_factory=list)


class AdaptationTaskList(StrictModel):
    """Everything still needed to produce a complete adaptation output set.

    `target_model_id` stays the bare platform id; `target_invocation_model_id`
    (additive) is the selector-qualified id deliverables must reference when a
    selector was chosen.
    """

    schema_version: Literal["3"] = "3"
    run_id: str
    source_model: str
    target_model: str
    target_model_id: str
    target_invocation_model_id: str | None = None
    target_invocation_selector: str | None = None
    prompt_tasks: list[PromptAdaptationTask] = Field(default_factory=list)
    file_tasks: list[FileAdaptationTask] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    shared_prompt_guidance: list[GuidanceItem] = Field(default_factory=list)


class PromptSubmissionResult(StrictModel):
    accepted: bool
    source_path: str
    output_path: str | None = None
    validation: PromptValidationResult
    message: str


class FileSubmissionResult(StrictModel):
    accepted: bool
    source_path: str
    output_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    message: str


class MigrationRunFinalization(StrictModel):
    """Summary of the finalized run output set.

    Blocker state is reported honestly in three buckets: blockers still
    unresolved, blockers resolved by a recorded user decision (with the
    decision), and recorded decisions that no longer match a live blocker.
    Deliverables that were reviewed and needed no change are counted in
    `reviewed_unchanged`, never inflated into the adapted counts.
    """

    schema_version: Literal["4"] = "4"
    run_id: str
    manifest_path: str
    report_path: str
    migration_complexity: str
    unresolved_blockers: list[str] = Field(default_factory=list)
    resolved_blockers: list[str] = Field(default_factory=list)
    stale_decisions: list[str] = Field(default_factory=list)
    adapted_prompts: int
    adapted_files: int
    reviewed_unchanged: int = 0
    undecided_changes: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    message: str


def sanitize_run_id(value: str) -> str:
    """Force an arbitrary label into the run-id pattern deterministically."""
    text = re.sub(r"[^a-z0-9._-]+", "-", value.strip().casefold()).strip("._-")
    return text or "run"


def default_run_id(source_model: str, target_model: str, as_of: date) -> str:
    return sanitize_run_id(f"{source_model}-to-{target_model}-{as_of.isoformat()}")


def default_run_dir(application_root: Path, run_id: str) -> Path:
    base = application_root if application_root.is_dir() else application_root.parent
    return base / DEFAULT_RUNS_SUBDIR / run_id


def run_paths(run_dir: Path) -> MigrationRunPaths:
    output = run_dir / "output"
    return MigrationRunPaths(
        run_dir=str(run_dir),
        config_path=str(run_dir / RUN_CONFIG_FILENAME),
        request_path=str(run_dir / "request.yaml"),
        decisions_path=str(run_dir / "decisions.yaml"),
        research_dir=str(run_dir / "research"),
        review_dir=str(run_dir / "review"),
        output_dir=str(output),
        prompts_dir=str(output / "prompts"),
        files_dir=str(output / "files"),
        manifest_path=str(output / "migration-manifest.yaml"),
        report_path=str(output / "migration-report.md"),
        changes_path=str(output / CHANGES_FILENAME),
        change_decisions_path=str(run_dir / CHANGE_DECISIONS_FILENAME),
    )


def write_run_config(run_dir: Path, config: MigrationRunConfig) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(exist_ok=True)
    path = run_dir / RUN_CONFIG_FILENAME
    atomic_write_text(path, yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return path


def load_run_config(run_dir: Path) -> MigrationRunConfig:
    path = Path(run_dir) / RUN_CONFIG_FILENAME
    if not path.is_file():
        raise WorkspaceError(f"{path} does not exist; start the run with start_migration first")
    try:
        return MigrationRunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise WorkspaceError(f"invalid run configuration {path}: {exc}") from exc


def load_adaptation_log(run_dir: Path, run_id: str) -> AdaptationLog:
    path = Path(run_dir) / "output" / CHANGES_FILENAME
    if not path.is_file():
        return AdaptationLog(run_id=run_id)
    try:
        return AdaptationLog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise WorkspaceError(f"invalid adaptation log {path}: {exc}") from exc


def _save_adaptation_log(run_dir: Path, log: AdaptationLog) -> None:
    path = Path(run_dir) / "output" / CHANGES_FILENAME
    log = log.model_copy(update={"schema_version": "2"})
    atomic_write_text(path, yaml.safe_dump(log.model_dump(mode="json"), sort_keys=False))


def _upsert_entry(log: AdaptationLog, entry: AdaptationEntry) -> AdaptationLog:
    kept = [
        item
        for item in log.entries
        if not (item.kind == entry.kind and item.source_path == entry.source_path)
    ]
    return log.model_copy(update={"entries": [*kept, entry]})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def submission_copy_path(run_dir: Path, kind: str, relative: Path | str) -> Path:
    """Where the as-submitted content of one deliverable is kept.

    The deliverable under output/ is the EFFECTIVE content (review rejections
    regenerate it); this copy preserves the submission itself so decisions
    can be re-applied deterministically in any order.
    """
    subdir = "prompts" if kind == "prompt" else "files"
    return Path(run_dir) / "review" / "submissions" / subdir / Path(relative)


def _write_submission_copy(run_dir: Path, kind: str, relative: Path, content: str) -> None:
    atomic_write_text(submission_copy_path(run_dir, kind, relative), content)


def _application_base(config: MigrationRunConfig) -> Path:
    root = Path(config.application_root)
    return root if root.is_dir() else root.parent


def _safe_relative_path(config: MigrationRunConfig, source_path: str) -> Path:
    if not source_path.strip() or any(char in source_path for char in '<>*?"|'):
        raise WorkspaceError(f"source_path must be a real relative file path, got {source_path!r}")
    relative = Path(source_path)
    if relative.is_absolute():
        raise WorkspaceError(
            f"source_path must be relative to the application root, got {source_path!r}"
        )
    base = _application_base(config).resolve()
    resolved = (base / relative).resolve()
    try:
        return resolved.relative_to(base)
    except ValueError as exc:
        raise WorkspaceError(f"source_path {source_path!r} escapes the application root") from exc


def _advice_texts(items: list[MigrationAdvice]) -> list[str]:
    return [
        item.text + (f" (evidence: {item.evidence_urls[0]})" if item.evidence_urls else "")
        for item in items
    ]


_CURATION_FINDING_CATEGORIES = {
    "duplicated_requirements",
    "negative_wording",
    "explicit_chain_of_thought",
    "assistant_prefill",
    "json_only_prompting",
}


def _spec_guidance(spec: PromptMigrationSpec) -> list[str]:
    return [
        *_advice_texts(spec.requirements_to_preserve),
        *_advice_texts(spec.assumptions_to_reconsider),
        *_advice_texts(spec.instructions_needing_strengthening),
        *_advice_texts(spec.target_features_replacing_prompt_text),
        *_advice_texts(spec.reasoning_configuration),
        *_advice_texts(spec.structured_output),
    ]


def _structured_candidate(
    config: MigrationRunConfig, source_path: str, specs: list[PromptMigrationSpec]
) -> str:
    """Deterministic full-document candidate for a structured prompt source."""
    original_path = _application_base(config) / source_path
    format = STRUCTURED_SUFFIXES.get(Path(source_path).suffix.casefold())
    try:
        original = original_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return "\n\n".join(spec.candidate_prompt for spec in specs)
    if format is None:
        return original
    replacements = {
        spec.source_component: spec.candidate_prompt for spec in specs if spec.source_component
    }
    try:
        rebuilt = build_candidate_document(original, format, replacements)
    except ValueError:
        rebuilt = None
    return rebuilt if rebuilt is not None else original


def derive_adaptation_tasks(
    config: MigrationRunConfig,
    plan: MigrationPlan,
    run_dir: Path,
    log: AdaptationLog | None = None,
) -> AdaptationTaskList:
    """Turn one migration plan into an explicit, per-file adaptation worklist.

    Every guidance and required-change text is invocation-qualified: bare
    target model ids the plan derived are rewritten to the run's chosen
    invocation id, so the toolkit never instructs an id the platform cannot
    invoke on demand.
    """
    if log is None:
        log = load_adaptation_log(run_dir, config.run_id)

    def invocation_qualified(text: str) -> str:
        return qualify_bare_references(
            text, config.target_model_id, target_spellings(config), effective_target_id(config)
        )

    submitted = {(entry.kind, entry.source_path) for entry in log.entries}
    prompt_tasks: list[PromptAdaptationTask] = []
    prompt_paths: set[str] = set()
    specs_by_path: dict[str, list[PromptMigrationSpec]] = {}
    for spec in plan.prompt_changes:
        if spec.source_path is None:
            continue  # No real path means no submittable (or closable) task.
        specs_by_path.setdefault(spec.source_path, []).append(spec)
    difference_guidance = [
        invocation_qualified(
            f"Model difference ({item.severity.value}): {item.migration_impact}"
            + (f" Action: {item.recommended_action}" if item.recommended_action else "")
            + (f" (evidence: {item.target_evidence_url})" if item.target_evidence_url else "")
        )
        for item in plan.model_differences.differences
        if item.category == "migration_knowledge" and item.severity is not ComparisonSeverity.INFO
    ]
    for source_path, specs in sorted(specs_by_path.items()):
        prompt_paths.add(source_path)
        components = [spec.source_component for spec in specs if spec.source_component]
        structured = bool(components)
        guidance: list[str] = []
        risks: list[str] = []
        if structured:
            guidance.append(
                "This is a structured prompt document. Adapt only the prompt component "
                f"value(s) {', '.join(components)}; preserve every other key and value "
                "exactly, and submit the complete rebuilt document."
            )
        for spec in specs:
            prefix = f"[{spec.source_component}] " if spec.source_component else ""
            sections = structural_sections(spec.candidate_prompt)
            if sections:
                guidance.append(
                    f"{prefix}Preserve the structural section(s) "
                    + ", ".join(f"<{name}>" for name in sections)
                    + "; submissions that drop them are rejected unless allow_restructure "
                    "is set with recorded evidence."
                )
            for finding in spec.source_prompt_analysis.findings:
                if finding.category not in _CURATION_FINDING_CATEGORIES:
                    continue
                evidence = f" Evidence: {finding.evidence[0]!r}." if finding.evidence else ""
                guidance.append(
                    f"{prefix}In-prompt finding ({finding.category}): {finding.message}{evidence}"
                )
            for text in _spec_guidance(spec):
                line = invocation_qualified(prefix + text)
                if line not in guidance:
                    guidance.append(line)
            for text in _advice_texts(spec.migration_risks):
                line = invocation_qualified(prefix + text)
                if line not in risks:
                    risks.append(line)
        candidate = (
            _structured_candidate(config, source_path, specs)
            if structured
            else specs[0].candidate_prompt
        )
        prompt_tasks.append(
            PromptAdaptationTask(
                source_path=source_path,
                output_path=str(Path("output") / "prompts" / source_path),
                status=("submitted" if ("prompt", source_path) in submitted else "pending"),
                verbatim_source=candidate,
                components=components,
                guidance=[guidance_item(line) for line in guidance],
                risks=risks,
            )
        )
    changes_by_file: dict[str, list[str]] = {}
    for change in (
        *plan.required_changes,
        *plan.tool_changes,
        *plan.output_contract_changes,
        *plan.configuration_changes,
    ):
        for file in change.files or ["<application>"]:
            changes_by_file.setdefault(file, []).append(invocation_qualified(change.description))
    file_tasks = [
        FileAdaptationTask(
            source_path=file,
            output_path=str(Path("output") / "files" / file),
            status=("submitted" if ("file", file) in submitted else "pending"),
            required_changes=[
                guidance_item(description) for description in sorted(set(descriptions))
            ],
        )
        for file, descriptions in sorted(changes_by_file.items())
        if file not in prompt_paths and file != "<application>"
    ]
    covered = {task.source_path for task in file_tasks} | prompt_paths
    for file in plan.affected_files:
        if file in covered:
            continue
        file_tasks.append(
            FileAdaptationTask(
                source_path=file,
                output_path=str(Path("output") / "files" / file),
                status=("submitted" if ("file", file) in submitted else "pending"),
                required_changes=[
                    guidance_item(
                        "Update every detected coupling in this file for "
                        f"{config.target.model} on {config.target.platform} "
                        f"(invocation model id {effective_target_id(config)})."
                    )
                ],
            )
        )
    file_tasks.sort(key=lambda task: task.source_path)
    # Required changes that target a prompt file (e.g. a decision-injected
    # prompt-reduction task) belong on that prompt task's guidance — prompt
    # paths are excluded from file tasks, and dropping them would lose a
    # REQUIRED change between the plan and the worklist.
    prompt_tasks = [
        (
            task.model_copy(
                update={
                    "guidance": [
                        *task.guidance,
                        *(
                            guidance_item(f"Required change: {description}")
                            for description in sorted(set(changes_by_file[task.source_path]))
                        ),
                    ]
                }
            )
            if task.source_path in changes_by_file
            else task
        )
        for task in prompt_tasks
    ]
    # Hoist guidance shared by every prompt task into one shared list, so the
    # task payload the host agent reads does not repeat it per task.
    shared_prompt_guidance = [guidance_item(line) for line in difference_guidance]
    if len(prompt_tasks) > 1:
        shared = [
            line
            for line in prompt_tasks[0].guidance
            if all(line in task.guidance for task in prompt_tasks[1:])
        ]
        if shared:
            shared_set = set(shared)
            prompt_tasks = [
                task.model_copy(
                    update={"guidance": [line for line in task.guidance if line not in shared_set]}
                )
                for task in prompt_tasks
            ]
            shared_prompt_guidance = [*shared, *shared_prompt_guidance]
    application_level_changes = [
        f"Plan-level required change (no specific file): {description}"
        for description in sorted(set(changes_by_file.get("<application>", [])))
    ]
    return AdaptationTaskList(
        run_id=config.run_id,
        source_model=config.source.model,
        target_model=config.target.model,
        target_model_id=config.target_model_id,
        target_invocation_model_id=config.target_invocation_model_id,
        target_invocation_selector=config.target_invocation_selector,
        prompt_tasks=prompt_tasks,
        file_tasks=file_tasks,
        blockers=[blocker.rendered for blocker in plan.blockers],
        guidance=[
            *application_level_changes,
            "Read each source file from the application, produce the complete adapted "
            "version, and submit it with submit_adapted_file; submit rewritten prompts "
            "with submit_adapted_prompt.",
            "Each prompt task's `verbatim_source` is the ORIGINAL unmodified content, "
            "never a proposed adaptation: produce the adapted version yourself from "
            "it and the guidance, and never present it to the user as the tool's "
            "suggestion.",
            "Adapt prompts minimally and only with evidence: keep the original wording "
            "and structure except where a listed model difference or evidence-linked "
            "guidance item requires a change, and record each edit in "
            "`annotated_changes` with the evidence that motivated it. "
            "`shared_prompt_guidance` applies to every prompt "
            "task. Structural drops (removed XML-like sections or components) are "
            "rejected unless the submission sets allow_restructure and records the "
            "justification.",
            "Every prompt submission must dispose EVERY guidance item of its task "
            "(the task's `guidance` plus `shared_prompt_guidance`) by id in "
            "`guidance_dispositions`: applied, not_applicable, or declined (declined "
            "requires a note). File submissions dispose their task's "
            "`required_changes` the same way. Submissions with missing, unknown, or "
            "duplicate dispositions are rejected, and an unchanged=true submission "
            "cannot claim any item as applied.",
            "Every CHANGED submission (prompt or file) must record its edits as "
            "`annotated_changes`: each entry anchors an exact text span of the "
            "original and/or adapted content (operation edit/insert/delete/"
            "restructure), explains in one sentence `why` the target model needs "
            "it, and cites `evidence` (a plan-carried URL or a reference; kind "
            "'mechanical' for typo-level fixes needs no citation). Every diff hunk "
            "must be covered by an annotation and every annotation must match a "
            "real edit — undocumented or phantom changes are rejected. Anchors are "
            "matched against the DECODED runtime values.",
            "Prompt submissions must change the DECODED runtime prompt values; "
            "serialization-only, whitespace-only, and case-only edits are rejected, "
            "and validation runs on the decoded values, so encoding tricks cannot "
            "clear a finding.",
            "If a prompt or file genuinely needs no change for the target model, "
            "submit it with unchanged=true (both submission tools support it) "
            "instead of inventing an edit or leaving a coverage gap; the final "
            "report then states explicitly that no change was needed and why.",
            "Call this worklist once and work through it; each submission result "
            "already confirms acceptance, and finalize_migration reports any "
            "remaining gaps, so there is no need to re-list between submissions.",
            "If `blockers` is non-empty, call get_blocker_resolutions(run_dir) and "
            "present each blocker's question, options, and evidence VERBATIM to the "
            "user, one blocker at a time; record each user answer with "
            "record_blocker_decision. Blockers come from the plan, not from your "
            "submissions — never retry submissions to make one disappear, and never "
            "choose an option on the user's behalf.",
            "Every submission must be the full finalized file content, not a diff.",
            "State in `changes` what was changed and in `rationale` why the target model "
            "needs it; both appear verbatim in the final report.",
            "Never edit the application tree directly; deliverables are written only "
            "beneath the run's output/ directory.",
        ],
        shared_prompt_guidance=shared_prompt_guidance,
    )


def read_original_prompt(config: MigrationRunConfig, source_path: str) -> str | None:
    """The application's current content for one prompt source path, if present."""
    original = _application_base(config) / _safe_relative_path(config, source_path)
    return original.read_text(encoding="utf-8") if original.is_file() else None


def structured_submission_format(
    config: MigrationRunConfig, source_path: str
) -> PromptSourceFormat | None:
    """The structured format of a submission path, resolved the one true way.

    Both the service-level validation and the workspace checks must derive the
    format from the same sanitized path, or a symlinked/odd path could be
    validated as plain text while being checked as a structured document.
    """
    return STRUCTURED_SUFFIXES.get(_safe_relative_path(config, source_path).suffix.casefold())


def _unchanged_prompt_problems(
    config: MigrationRunConfig,
    original_text: str,
    format: PromptSourceFormat | None,
    rationale: str,
) -> list[str]:
    """Guards for recording a prompt as reviewed-but-unchanged."""
    problems: list[str] = []
    if not rationale.strip():
        problems.append("unchanged=true requires a rationale recording the review")
    target_forms = {*target_spellings(config), config.target.model}
    needles = [
        needle
        for needle in (*source_spellings(config), config.source.model)
        if needle and needle not in target_forms
    ]
    mentioned = sorted(
        {needle for needle in needles if document_mentions(original_text, format, needle)}
    )
    if mentioned:
        problems.append(
            "the prompt's decoded values reference the source model ("
            + ", ".join(mentioned)
            + "); it cannot be recorded as unchanged"
        )
    return problems


def _shorten(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


_FORMAT_REQUIREMENT_NOTE = (
    " NOTE: these rejection(s) are submission-format requirements of this tool "
    "(guidance dispositions / annotated changes recorded in the submission "
    "payload), not a judgement on the adaptation content. Fix the submission "
    "fields and resubmit; do not ask the user to refine the prompt or the "
    "adaptation."
)


def _disposition_problems(
    required: list[GuidanceItem],
    dispositions: list[GuidanceDisposition],
    unchanged: bool,
) -> list[str]:
    """Fail-closed reconciliation of dispositions against the task's guidance.

    Every guidance item must be disposed exactly once, a decline must record
    why, and an unchanged submission cannot claim an item as applied — an
    applied item implies an edit.
    """
    problems: list[str] = []
    known = {item.id: item.text for item in required}
    seen: set[str] = set()
    for disposition in dispositions:
        if disposition.guidance_id not in known:
            problems.append(
                f"disposition references unknown guidance id {disposition.guidance_id!r}; "
                "ids come from the task's `guidance` and the worklist's "
                "`shared_prompt_guidance`"
            )
            continue
        if disposition.guidance_id in seen:
            problems.append(f"guidance id {disposition.guidance_id!r} is disposed more than once")
            continue
        seen.add(disposition.guidance_id)
        if disposition.disposition == "declined" and not disposition.note.strip():
            problems.append(
                f"declining guidance {disposition.guidance_id!r} requires a note "
                "recording why it was deliberately not applied"
            )
        if unchanged and disposition.disposition == "applied":
            problems.append(
                f"guidance {disposition.guidance_id!r} is marked applied, which "
                "contradicts unchanged=true; an applied item implies an edit"
            )
    missing = [item for item in required if item.id not in seen]
    if missing:
        owed = "; ".join(f"{item.id} ({_shorten(item.text)})" for item in missing)
        problems.append(
            "every guidance item must be disposed as applied, not_applicable, or "
            f"declined in `guidance_dispositions`; still owed: {owed}"
        )
    return problems


def submit_adapted_prompt(
    run_dir: Path,
    config: MigrationRunConfig,
    source_path: str,
    adapted_prompt: str,
    rationale: str,
    changes: list[str],
    validation: PromptValidationResult,
    submitted_on: date,
    allow_restructure: bool = False,
    unchanged: bool = False,
    guidance_dispositions: list[GuidanceDisposition] | None = None,
    tasks: AdaptationTaskList | None = None,
    annotated_changes: list[AnnotatedChange] | None = None,
    known_evidence_urls: set[str] | None = None,
) -> PromptSubmissionResult:
    """Persist one validated adapted prompt beneath output/prompts/.

    `unchanged=True` records that the prompt was reviewed and needs no change:
    the original content is copied as the deliverable. Without it, a
    submission whose DECODED runtime prompt values equal the original's is
    rejected — byte-level or serialization changes (escapes, quoting,
    whitespace style) are not an adaptation and cannot clear validation.
    When `tasks` is provided and lists this prompt, `guidance_dispositions`
    must reconcile every guidance item of the task (see
    `_disposition_problems`), and a changed submission must reconcile its
    `annotated_changes` against the real diff of the decoded runtime values
    (see `annotation_problems`).
    """
    relative = _safe_relative_path(config, source_path)
    blockers = [
        issue.message for issue in validation.issues if issue.level is ValidationLevel.BLOCKER
    ]
    dispositions = list(guidance_dispositions or [])
    annotations = list(annotated_changes or [])
    required_guidance: list[GuidanceItem] = []
    if tasks is not None:
        task = next(
            (item for item in tasks.prompt_tasks if item.source_path == relative.as_posix()),
            None,
        )
        if task is not None:
            required_guidance = [*task.guidance, *tasks.shared_prompt_guidance]
    format_problems = _disposition_problems(required_guidance, dispositions, unchanged)
    if unchanged and annotations:
        format_problems.append(
            "unchanged=true cannot carry annotated_changes; there is no edit to annotate"
        )
    blockers.extend(format_problems)
    original = _application_base(config) / relative
    original_text = original.read_text(encoding="utf-8") if original.is_file() else None
    format = STRUCTURED_SUFFIXES.get(relative.suffix.casefold())
    structure_warnings: list[str] = []
    if unchanged:
        if original_text is None:
            return PromptSubmissionResult(
                accepted=False,
                source_path=source_path,
                validation=validation,
                message=(
                    "Rejected: unchanged=true requires the prompt file to exist in the application"
                ),
            )
        problems = _unchanged_prompt_problems(config, original_text, format, rationale)
        if problems:
            return PromptSubmissionResult(
                accepted=False,
                source_path=source_path,
                validation=validation,
                message="Rejected: " + "; ".join(problems),
            )
        adapted_prompt = original_text
        if not changes:
            changes = ["Reviewed for the target model; no change required."]
    elif original_text is None:
        structure_warnings.append(
            "the original prompt file was not found in the application; document and "
            "structural checks were skipped and no source fingerprint was recorded"
        )
    elif adapted_prompt.strip():
        assessment = evaluate_prompt_submission(
            original_text,
            adapted_prompt,
            format,
            source_model_ids=source_spellings(config),
            target_model_ids=target_spellings(config),
        )
        blockers.extend(assessment.problems)
        if assessment.checks_skipped:
            structure_warnings.append(assessment.checks_skipped)
        clean = not assessment.problems and not assessment.checks_skipped
        if clean and not assessment.runtime_changed:
            blockers.append(
                "the adapted prompt decodes to the same runtime values as the original; "
                "byte-level or serialization changes are not an adaptation. Resubmit with "
                "unchanged=true to record a reviewed no-change prompt"
            )
        elif clean and assessment.cosmetic_only:
            blockers.append(
                "the adaptation changes only whitespace or letter case in the runtime "
                "prompt values; that is not a target-model adaptation. Make a substantive "
                "change, or resubmit with unchanged=true to record a reviewed no-change "
                "prompt"
            )
        drops = assessment.structural_drops
        if drops and not allow_restructure:
            blockers.append(
                "the adapted prompt restructures the original without justification: "
                + "; ".join(drops)
                + ". Adapt minimally and preserve the original sections, or resubmit with "
                "allow_restructure=true and record the evidence for the restructure in `changes`"
            )
        elif drops and not changes:
            blockers.append(
                "allow_restructure requires at least one `changes` entry recording what "
                "was restructured and the evidence for it"
            )
        elif drops:
            structure_warnings.extend(f"restructure accepted: {drop}" for drop in drops)
        structure_warnings.extend(f"note: {note}" for note in assessment.notes)
        decoded_original = decoded_view(original_text, format)
        decoded_adapted = decoded_view(adapted_prompt, format)
        if decoded_original is None or decoded_adapted is None:
            if annotations:
                structure_warnings.append(
                    "annotated_changes could not be checked against the diff because "
                    "the document could not be decoded"
                )
        else:
            annotation_issues, annotation_warnings = annotation_problems(
                decoded_original,
                decoded_adapted,
                annotations,
                known_evidence_urls=known_evidence_urls,
            )
            blockers.extend(annotation_issues)
            format_problems.extend(annotation_issues)
            structure_warnings.extend(annotation_warnings)
    if not adapted_prompt.strip():
        blockers.append("the adapted prompt is empty")
    selector_ids = [item for item in target_spellings(config) if item != config.target_model_id]
    if (
        not unchanged
        and config.target_invocation_requires_selector
        and selector_ids
        and adapted_prompt.strip()
    ):
        # Check the raw text AND the decoded values, so serialization tricks
        # cannot hide a forbidden bare-id reference (same gate as file
        # submissions: the reviewed profile says the bare id is not invocable).
        views = [adapted_prompt]
        decoded = decoded_view(adapted_prompt, format)
        if decoded is not None:
            views.append(decoded)
        if any(references_bare_alone(view, config.target_model_id, selector_ids) for view in views):
            blockers.append(
                f"the adapted prompt references the bare platform model id "
                f"{config.target_model_id!r}, which the reviewed profile states is not "
                "invocable on demand; reference an invocation selector id instead: "
                + ", ".join(selector_ids)
            )
    if blockers:
        return PromptSubmissionResult(
            accepted=False,
            source_path=source_path,
            validation=validation,
            message="Rejected: "
            + "; ".join(blockers)
            + (_FORMAT_REQUIREMENT_NOTE if format_problems else ""),
        )
    source_sha = _sha256(original_text) if original_text is not None else None
    output_path = Path(run_dir) / "output" / "prompts" / relative
    guidance_texts = {item.id: item.text for item in required_guidance}
    entry = AdaptationEntry(
        kind="prompt",
        source_path=relative.as_posix(),
        output_path=str(Path("output") / "prompts" / relative),
        rationale=rationale,
        changes=changes,
        warnings=[
            *(
                issue.message
                for issue in validation.issues
                if issue.level is ValidationLevel.WARNING
            ),
            *structure_warnings,
        ],
        source_sha256=source_sha,
        adapted_sha256=_sha256(adapted_prompt),
        submitted_on=submitted_on,
        unchanged=unchanged,
        guidance_dispositions=[
            item.model_copy(update={"guidance": guidance_texts.get(item.guidance_id, "")})
            for item in dispositions
        ],
        annotated_changes=with_change_ids(annotations),
    )
    with run_state_lock(run_dir):
        atomic_write_text(output_path, adapted_prompt)
        _write_submission_copy(run_dir, "prompt", relative, adapted_prompt)
        _save_adaptation_log(
            run_dir, _upsert_entry(load_adaptation_log(run_dir, config.run_id), entry)
        )
    return PromptSubmissionResult(
        accepted=True,
        source_path=source_path,
        output_path=str(output_path),
        validation=validation,
        message=f"Adapted prompt written to {output_path}."
        + (
            f" Accepted with {len(structure_warnings)} recorded warning(s): "
            + "; ".join(structure_warnings)
            if structure_warnings
            else ""
        ),
    )


def submit_adapted_file(
    run_dir: Path,
    config: MigrationRunConfig,
    source_path: str,
    adapted_content: str,
    rationale: str,
    changes: list[str],
    submitted_on: date,
    new_file: bool = False,
    unchanged: bool = False,
    guidance_dispositions: list[GuidanceDisposition] | None = None,
    tasks: AdaptationTaskList | None = None,
    annotated_changes: list[AnnotatedChange] | None = None,
    known_evidence_urls: set[str] | None = None,
) -> FileSubmissionResult:
    """Persist one adapted application file beneath output/files/, fail closed.

    `unchanged=True` records that the file was reviewed and needs no change
    for the target model: the original content is copied as the deliverable
    (any `adapted_content` is ignored), closing the coverage gap without
    forcing an invented edit. When `tasks` is provided and lists this file,
    `guidance_dispositions` must dispose every required change of the task,
    and a changed submission of an existing file must reconcile its
    `annotated_changes` against the real diff (see `annotation_problems`).
    """
    relative = _safe_relative_path(config, source_path)
    original_path = _application_base(config) / relative
    problems: list[str] = []
    warnings: list[str] = []
    original: str | None = None
    if original_path.is_file():
        original = original_path.read_text(encoding="utf-8")
    elif not new_file:
        problems.append(
            f"{relative.as_posix()} does not exist in the application; pass new_file=true "
            "only when the migration genuinely introduces a new file"
        )
    dispositions = list(guidance_dispositions or [])
    annotations = list(annotated_changes or [])
    required_changes: list[GuidanceItem] = []
    if tasks is not None:
        task = next(
            (item for item in tasks.file_tasks if item.source_path == relative.as_posix()),
            None,
        )
        if task is not None:
            required_changes = list(task.required_changes)
    format_problems = _disposition_problems(required_changes, dispositions, unchanged)
    if unchanged and annotations:
        format_problems.append(
            "unchanged=true cannot carry annotated_changes; there is no edit to annotate"
        )
    elif not unchanged and original is not None and adapted_content.strip():
        annotation_issues, annotation_warnings = annotation_problems(
            original,
            adapted_content,
            annotations,
            known_evidence_urls=known_evidence_urls,
        )
        format_problems.extend(annotation_issues)
        warnings.extend(annotation_warnings)
    elif annotations and original is None:
        standalone_problems, standalone_warnings = standalone_annotation_problems(
            annotations, known_evidence_urls
        )
        format_problems.extend(standalone_problems)
        warnings.extend(standalone_warnings)
        warnings.append(
            "annotated_changes could not be checked against a diff because the file "
            "has no original in the application"
        )
    unchanged_source_mention = (
        next(
            (
                spelling
                for spelling in source_spellings(config)
                if spelling not in set(target_spellings(config)) and spelling in (original or "")
            ),
            None,
        )
        if unchanged
        else None
    )
    if unchanged:
        if original is None:
            problems.append("unchanged=true requires the file to exist in the application")
        elif unchanged_source_mention is not None:
            problems.append(
                f"the file references the source model id {unchanged_source_mention!r}; "
                "it cannot be recorded as unchanged"
            )
        else:
            adapted_content = original
            if not changes:
                changes = ["Reviewed for the target model; no change required."]
    if not adapted_content.strip():
        problems.append("the adapted file content is empty")
    if not unchanged and original is not None and original == adapted_content:
        problems.append(
            "the adapted content is identical to the source file; pass unchanged=true "
            "if this file genuinely needs no change"
        )
    prompt_format = source_format(relative.as_posix())
    if (
        original is not None
        and prompt_format is not None
        and is_prompt_bearing(original, prompt_format)
    ):
        problems.append(
            f"{relative.as_posix()} is a prompt source; submit it with "
            "submit_adapted_prompt so target-model validation and structural "
            "checks apply"
        )
    if relative.suffix == ".py" and not unchanged:
        try:
            ast.parse(adapted_content)
        except SyntaxError as exc:
            problems.append(f"the adapted Python content does not parse: {exc}")
    selector_ids = [item for item in target_spellings(config) if item != config.target_model_id]
    if (
        not unchanged
        and config.target_invocation_requires_selector
        and selector_ids
        and references_bare_alone(adapted_content, config.target_model_id, selector_ids)
    ):
        problems.append(
            f"the adapted content references the bare platform model id "
            f"{config.target_model_id!r}, which the reviewed profile states is not "
            "invocable on demand; reference an invocation selector id instead: "
            + ", ".join(selector_ids)
        )
    if not changes and not annotations:
        problems.append(
            "at least one annotated change (or `changes` entry) describing the edit is required"
        )
    problems.extend(format_problems)
    if problems:
        return FileSubmissionResult(
            accepted=False,
            source_path=source_path,
            problems=problems,
            message="Rejected: "
            + "; ".join(problems)
            + (_FORMAT_REQUIREMENT_NOTE if format_problems else ""),
        )
    target_forms = target_spellings(config)
    source_forms = [item for item in source_spellings(config) if item not in set(target_forms)]
    remaining = [item for item in source_forms if item in adapted_content]
    # Report the most specific spelling only (a bare id is a substring of its
    # selector-qualified forms and would double-report).
    for spelling in remaining:
        if any(spelling != other and spelling in other for other in remaining):
            continue
        warnings.append(f"the source model id {spelling!r} still appears in the adapted content")
    if (
        original is not None
        and any(item in original for item in source_forms)
        and not any(item in adapted_content for item in target_forms)
    ):
        warnings.append(
            "the original file referenced the source model id but the adapted "
            f"content never references the target invocation model id "
            f"{effective_target_id(config)!r}"
        )
    output_path = Path(run_dir) / "output" / "files" / relative
    required_texts = {item.id: item.text for item in required_changes}
    entry = AdaptationEntry(
        kind="file",
        source_path=relative.as_posix(),
        output_path=str(Path("output") / "files" / relative),
        rationale=rationale,
        changes=changes,
        warnings=warnings,
        source_sha256=_sha256(original) if original is not None else None,
        adapted_sha256=_sha256(adapted_content),
        submitted_on=submitted_on,
        unchanged=unchanged,
        guidance_dispositions=[
            item.model_copy(update={"guidance": required_texts.get(item.guidance_id, "")})
            for item in dispositions
        ],
        annotated_changes=with_change_ids(annotations),
    )
    with run_state_lock(run_dir):
        atomic_write_text(output_path, adapted_content)
        _write_submission_copy(run_dir, "file", relative, adapted_content)
        _save_adaptation_log(
            run_dir, _upsert_entry(load_adaptation_log(run_dir, config.run_id), entry)
        )
    return FileSubmissionResult(
        accepted=True,
        source_path=source_path,
        output_path=str(output_path),
        warnings=warnings,
        message=f"Adapted file written to {output_path}.",
    )


def coverage_gaps(tasks: AdaptationTaskList) -> list[str]:
    """Worklist entries that still have no submitted adaptation deliverable.

    Computed from the same derived task list the agent works through, so the
    worklist and the finalization coverage report can never disagree about
    which files count.
    """
    pending: list[PromptAdaptationTask | FileAdaptationTask] = [
        *tasks.prompt_tasks,
        *tasks.file_tasks,
    ]
    return sorted(task.source_path for task in pending if task.status == "pending")


_DISPOSITION_LABELS = {
    "applied": "applied",
    "not_applicable": "not applicable",
    "declined": "declined",
}

_UNCHANGED_DEFAULT_NOTE = "Reviewed for the target model; no change required."


def _render_evidence(change: AnnotatedChange) -> str:
    parts = []
    for entry in change.evidence:
        detail = " ".join(part for part in (entry.url, entry.reference) if part)
        parts.append(entry.kind + (f" {detail}" if detail else ""))
    return "; ".join(parts) if parts else "none recorded"


def _render_annotated_changes(
    entry: AdaptationEntry, decisions: dict[str, ChangeDecision] | None = None
) -> list[str]:
    """One line of why-plus-evidence per change, with its before/after spans.

    With a decision index, each change carries its review status and an
    all-rejected entry states that the deliverable reverted to the original.
    """
    lines = ["- What changed:"]
    counts = {"accepted": 0, "rejected": 0, "pending": 0}
    for change in entry.annotated_changes:
        decision = decisions.get(change.id) if decisions is not None else None
        status = ""
        if decisions is not None:
            if decision is None:
                status = " — pending review"
                counts["pending"] += 1
            elif decision.decision == "accepted":
                status = " — ACCEPTED in review"
                counts["accepted"] += 1
            else:
                note = f" (note: {decision.note})" if decision.note.strip() else ""
                status = f" — REJECTED in review, reverted{note}"
                counts["rejected"] += 1
        lines.append(
            f"  - [{change.id}] {change.why} ({change.operation}; "
            f"evidence: {_render_evidence(change)}){status}"
        )
        if change.original_anchor and change.original_anchor.strip():
            lines.append(f"    - before: {_shorten(change.original_anchor, 200)!r}")
        if change.adapted_anchor and change.adapted_anchor.strip():
            lines.append(f"    - after: {_shorten(change.adapted_anchor, 200)!r}")
    if decisions is not None and (counts["accepted"] or counts["rejected"]):
        if counts["rejected"] == len(entry.annotated_changes) and entry.annotated_changes:
            lines.append(
                "- Review outcome: every change was rejected; no annotated adaptation "
                "remains in the deliverable."
            )
        else:
            lines.append(
                f"- Review outcome: {counts['accepted']} accepted, "
                f"{counts['rejected']} rejected, {counts['pending']} pending."
            )
    if entry.changes:
        lines.append("- Notes:")
        lines.extend(f"  - {change}" for change in entry.changes)
    return lines


def _render_dispositions(entry: AdaptationEntry) -> list[str]:
    if not entry.guidance_dispositions:
        return []
    lines = ["- Guidance dispositions:"]
    for item in entry.guidance_dispositions:
        text = item.guidance or item.guidance_id
        note = f" (note: {item.note})" if item.note.strip() else ""
        lines.append(f"  - {_DISPOSITION_LABELS[item.disposition]} — {text}{note}")
    return lines


def render_adaptation_section(
    log: AdaptationLog,
    gaps: list[str],
    decision_log: ChangeDecisionLog | None = None,
) -> str:
    """Render the per-file adaptation changes and rationale for the report.

    With a decision log, each annotated change carries its review status;
    only decisions that still match the entry's submission fingerprints are
    applied (stale ones are ignored here and surfaced by the review surface).
    """
    lines = [
        "## Adaptation deliverables",
        "",
        "Finalized post-adaptation files live beneath this run's `output/` directory. "
        "They are review candidates: apply each one to the application only after "
        "reviewing it, and never let this tooling overwrite the application tree.",
        "",
    ]
    if not log.entries:
        lines.append("- No adaptation deliverables were submitted for this run.")
    else:
        adapted_count = sum(not entry_is_unchanged(entry) for entry in log.entries)
        unchanged_count = len(log.entries) - adapted_count
        lines.extend(
            (
                f"{adapted_count} file(s) adapted; {unchanged_count} reviewed with "
                "no change needed.",
                "",
            )
        )
    for entry in sorted(log.entries, key=lambda item: (item.kind, item.source_path)):
        entry_decisions: dict[str, ChangeDecision] | None = None
        if decision_log is not None:
            entry_decisions = {
                decision.change_id: decision
                for decision in decision_log.decisions
                if decision_matches_entry(decision, entry)
            }
        if entry_is_unchanged(entry):
            review_notes = [change for change in entry.changes if change != _UNCHANGED_DEFAULT_NOTE]
            lines.extend(
                (
                    f"### `{entry.source_path}` ({entry.kind}) — no change needed",
                    "",
                    "- Reviewed for the target model; no adaptation was required. "
                    "The deliverable is the original content.",
                    f"- Why no change: {entry.rationale}",
                )
            )
            if review_notes:
                lines.append("- Review notes:")
                lines.extend(f"  - {note}" for note in review_notes)
        else:
            lines.extend(
                (
                    f"### `{entry.source_path}` ({entry.kind})",
                    "",
                    f"- Adapted file: `{entry.output_path}`",
                    f"- Why: {entry.rationale}",
                )
            )
            if entry.annotated_changes:
                lines.extend(_render_annotated_changes(entry, entry_decisions))
            else:
                lines.extend(
                    (
                        "- What changed:",
                        *(
                            [f"  - {change}" for change in entry.changes]
                            or ["  - (no change descriptions were recorded)"]
                        ),
                    )
                )
        lines.extend(_render_dispositions(entry))
        if entry.warnings:
            lines.append("- Validation warnings:")
            lines.extend(f"  - {warning}" for warning in entry.warnings)
        lines.append("")
    lines.extend(("### Adaptation coverage", ""))
    if gaps:
        submitted_kinds = {entry.source_path: entry.kind for entry in log.entries}
        for file in gaps:
            other_kind = submitted_kinds.get(file)
            if other_kind is not None:
                lines.append(
                    f"- `{file}`: a {other_kind} deliverable exists for this path, but the "
                    "task requires the other submission kind (prompt sources must go "
                    "through submit_adapted_prompt); the required checks were bypassed."
                )
            else:
                lines.append(
                    f"- `{file}`: no adapted version was submitted; its planned changes "
                    "remain manual."
                )
    else:
        lines.append("- Every affected file has a submitted adaptation deliverable.")
    lines.append("")
    return "\n".join(lines)
