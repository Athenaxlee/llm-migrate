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

from llm_migrate.analyzers.prompt import analyze_prompt
from llm_migrate.core.agent_research import RUN_ID_PATTERN, ModelEndpointIdentity
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
    PromptDocumentError,
    build_candidate_document,
    extract_prompt_components,
    parse_structured_document,
    structured_submission_problems,
)

RUN_CONFIG_FILENAME = "migration.yaml"
CHANGES_FILENAME = "changes.yaml"
DEFAULT_RUNS_SUBDIR = Path(".llm-migrate") / "runs"


class WorkspaceError(ValueError):
    """A run workspace operation cannot proceed as requested."""


class MigrationRunConfig(StrictModel):
    """Durable identity of one migration run, persisted as `migration.yaml`."""

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1, pattern=RUN_ID_PATTERN)
    application_root: str
    source: ModelEndpointIdentity
    target: ModelEndpointIdentity
    source_model_id: str
    target_model_id: str
    created_on: date
    prompt_sources: list[str] = Field(default_factory=list)


class MigrationRunPaths(StrictModel):
    """Where one run keeps its artifacts; all deliverables live under output/."""

    run_dir: str
    config_path: str
    request_path: str
    research_dir: str
    review_dir: str
    output_dir: str
    prompts_dir: str
    files_dir: str
    manifest_path: str
    report_path: str
    changes_path: str


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


class AdaptationLog(StrictModel):
    """Reviewable record of every adaptation deliverable in one run."""

    schema_version: Literal["1"] = "1"
    run_id: str
    entries: list[AdaptationEntry] = Field(default_factory=list)


class PromptAdaptationTask(StrictModel):
    """One prompt file the host agent must rewrite for the target model."""

    source_path: str
    output_path: str
    status: Literal["pending", "submitted"]
    deterministic_candidate: str
    components: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class FileAdaptationTask(StrictModel):
    """One application file the host agent must adapt for the target model."""

    source_path: str
    output_path: str
    status: Literal["pending", "submitted"]
    required_changes: list[str] = Field(default_factory=list)


class AdaptationTaskList(StrictModel):
    """Everything still needed to produce a complete adaptation output set."""

    schema_version: Literal["1"] = "1"
    run_id: str
    source_model: str
    target_model: str
    target_model_id: str
    prompt_tasks: list[PromptAdaptationTask] = Field(default_factory=list)
    file_tasks: list[FileAdaptationTask] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


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
    """Summary of the finalized run output set."""

    schema_version: Literal["1"] = "1"
    run_id: str
    manifest_path: str
    report_path: str
    migration_complexity: str
    blockers: list[str] = Field(default_factory=list)
    adapted_prompts: int
    adapted_files: int
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
        research_dir=str(run_dir / "research"),
        review_dir=str(run_dir / "review"),
        output_dir=str(output),
        prompts_dir=str(output / "prompts"),
        files_dir=str(output / "files"),
        manifest_path=str(output / "migration-manifest.yaml"),
        report_path=str(output / "migration-report.md"),
        changes_path=str(output / CHANGES_FILENAME),
    )


def write_run_config(run_dir: Path, config: MigrationRunConfig) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "output").mkdir(exist_ok=True)
    path = run_dir / RUN_CONFIG_FILENAME
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(log.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


def _upsert_entry(log: AdaptationLog, entry: AdaptationEntry) -> AdaptationLog:
    kept = [
        item
        for item in log.entries
        if not (item.kind == entry.kind and item.source_path == entry.source_path)
    ]
    return log.model_copy(update={"entries": [*kept, entry]})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _missing_tags(original: str, adapted: str) -> list[str]:
    original_tags = set(analyze_prompt(original).xml_like_tags)
    return sorted(original_tags - set(analyze_prompt(adapted).xml_like_tags))


def _structural_drops(
    original_text: str, adapted_text: str, format: PromptSourceFormat | None
) -> list[str]:
    """Structure the adapted prompt lost relative to the original.

    XML-like sections are deliberate prompt architecture; dropping or merging
    them is a restructure, not an adaptation, and needs explicit justification.
    """
    if format is not None:
        try:
            original_components = extract_prompt_components(
                parse_structured_document(original_text, format)
            )
            adapted_components = extract_prompt_components(
                parse_structured_document(adapted_text, format)
            )
        except PromptDocumentError:
            return []  # Syntax problems are reported by the document checks.
        adapted_by_key = {component.key: component for component in adapted_components}
        drops: list[str] = []
        for component in original_components:
            adapted = adapted_by_key.get(component.key)
            if adapted is None:
                drops.append(f"prompt component {component.key!r} was removed")
                continue
            missing = _missing_tags(component.content, adapted.content)
            if missing:
                drops.append(
                    f"{component.key!r} drops structural section(s) "
                    + ", ".join(f"<{tag}>" for tag in missing)
                )
        return drops
    missing = _missing_tags(original_text, adapted_text)
    if missing:
        return ["drops structural section(s) " + ", ".join(f"<{tag}>" for tag in missing)]
    return []


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
) -> AdaptationTaskList:
    """Turn one migration plan into an explicit, per-file adaptation worklist."""
    log = load_adaptation_log(run_dir, config.run_id)
    submitted = {(entry.kind, entry.source_path) for entry in log.entries}
    prompt_tasks: list[PromptAdaptationTask] = []
    prompt_paths: set[str] = set()
    specs_by_path: dict[str, list[PromptMigrationSpec]] = {}
    for spec in plan.prompt_changes:
        specs_by_path.setdefault(spec.source_path or "<inline>", []).append(spec)
    difference_guidance = [
        f"Model difference ({item.severity.value}): {item.migration_impact}"
        + (f" Action: {item.recommended_action}" if item.recommended_action else "")
        + (f" (evidence: {item.target_evidence_url})" if item.target_evidence_url else "")
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
            tags = spec.source_prompt_analysis.xml_like_tags
            if tags:
                label = f"[{spec.source_component}] " if spec.source_component else ""
                guidance.append(
                    f"{label}Preserve the structural section(s) "
                    + ", ".join(f"<{tag}>" for tag in tags)
                    + "; submissions that drop them are rejected unless allow_restructure "
                    "is set with recorded evidence."
                )
        guidance.extend(difference_guidance)
        for spec in specs:
            prefix = (
                f"[{spec.source_component}] " if spec.source_component and len(specs) > 1 else ""
            )
            for text in _spec_guidance(spec):
                line = prefix + text
                if line not in guidance:
                    guidance.append(line)
            for text in _advice_texts(spec.migration_risks):
                line = prefix + text
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
                deterministic_candidate=candidate,
                components=components,
                guidance=guidance,
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
            changes_by_file.setdefault(file, []).append(change.description)
    file_tasks = [
        FileAdaptationTask(
            source_path=file,
            output_path=str(Path("output") / "files" / file),
            status=("submitted" if ("file", file) in submitted else "pending"),
            required_changes=sorted(set(descriptions)),
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
                    "Update every detected coupling in this file for "
                    f"{config.target.model} on {config.target.platform} "
                    f"(model id {config.target_model_id})."
                ],
            )
        )
    file_tasks.sort(key=lambda task: task.source_path)
    return AdaptationTaskList(
        run_id=config.run_id,
        source_model=config.source.model,
        target_model=config.target.model,
        target_model_id=config.target_model_id,
        prompt_tasks=prompt_tasks,
        file_tasks=file_tasks,
        blockers=plan.blockers,
        guidance=[
            "Read each source file from the application, produce the complete adapted "
            "version, and submit it with submit_adapted_file; submit rewritten prompts "
            "with submit_adapted_prompt.",
            "Adapt prompts minimally and only with evidence: keep the original wording "
            "and structure except where a listed model difference or evidence-linked "
            "guidance item requires a change, and say in `changes` which evidence "
            "motivated each edit. Structural drops (removed XML-like sections or "
            "components) are rejected unless the submission sets allow_restructure and "
            "records the justification.",
            "Every submission must be the full finalized file content, not a diff.",
            "State in `changes` what was changed and in `rationale` why the target model "
            "needs it; both appear verbatim in the final report.",
            "Never edit the application tree directly; deliverables are written only "
            "beneath the run's output/ directory.",
        ],
    )


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
) -> PromptSubmissionResult:
    """Persist one validated adapted prompt beneath output/prompts/."""
    relative = _safe_relative_path(config, source_path)
    blockers = [
        issue.message for issue in validation.issues if issue.level is ValidationLevel.BLOCKER
    ]
    if not adapted_prompt.strip():
        blockers.append("the adapted prompt is empty")
    original = _application_base(config) / relative
    original_text = original.read_text(encoding="utf-8") if original.is_file() else None
    format = STRUCTURED_SUFFIXES.get(relative.suffix.casefold())
    structure_warnings: list[str] = []
    if original_text is not None:
        if format is not None:
            blockers.extend(structured_submission_problems(original_text, adapted_prompt, format))
        drops = _structural_drops(original_text, adapted_prompt, format)
        if drops and allow_restructure:
            structure_warnings = [f"restructure accepted: {drop}" for drop in drops]
        elif drops:
            blockers.append(
                "the adapted prompt restructures the original without justification: "
                + "; ".join(drops)
                + ". Adapt minimally and preserve the original sections, or resubmit with "
                "allow_restructure=true and record the evidence for the restructure in `changes`"
            )
    if blockers:
        return PromptSubmissionResult(
            accepted=False,
            source_path=source_path,
            validation=validation,
            message="Rejected: " + "; ".join(blockers),
        )
    source_sha = _sha256(original_text) if original_text is not None else None
    output_path = Path(run_dir) / "output" / "prompts" / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(adapted_prompt, encoding="utf-8")
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
    )
    _save_adaptation_log(run_dir, _upsert_entry(load_adaptation_log(run_dir, config.run_id), entry))
    return PromptSubmissionResult(
        accepted=True,
        source_path=source_path,
        output_path=str(output_path),
        validation=validation,
        message=f"Adapted prompt written to {output_path}.",
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
) -> FileSubmissionResult:
    """Persist one adapted application file beneath output/files/, fail closed."""
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
    if not adapted_content.strip():
        problems.append("the adapted file content is empty")
    if original is not None and original == adapted_content:
        problems.append("the adapted content is identical to the source file")
    if relative.suffix == ".py":
        try:
            ast.parse(adapted_content)
        except SyntaxError as exc:
            problems.append(f"the adapted Python content does not parse: {exc}")
    if not changes:
        problems.append("at least one `changes` entry describing the edit is required")
    if problems:
        return FileSubmissionResult(
            accepted=False,
            source_path=source_path,
            problems=problems,
            message="Rejected: " + "; ".join(problems),
        )
    if config.source_model_id != config.target_model_id:
        if config.source_model_id in adapted_content:
            warnings.append(
                f"the source model id {config.source_model_id!r} still appears in the "
                "adapted content"
            )
        if (
            original is not None
            and config.source_model_id in original
            and config.target_model_id not in adapted_content
        ):
            warnings.append(
                f"the original file referenced {config.source_model_id!r} but the adapted "
                f"content never references the target model id {config.target_model_id!r}"
            )
    output_path = Path(run_dir) / "output" / "files" / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(adapted_content, encoding="utf-8")
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
    )
    _save_adaptation_log(run_dir, _upsert_entry(load_adaptation_log(run_dir, config.run_id), entry))
    return FileSubmissionResult(
        accepted=True,
        source_path=source_path,
        output_path=str(output_path),
        warnings=warnings,
        message=f"Adapted file written to {output_path}.",
    )


def coverage_gaps(plan: MigrationPlan, log: AdaptationLog) -> list[str]:
    """Affected files that still have no submitted adaptation deliverable."""
    submitted = {entry.source_path for entry in log.entries}
    return [file for file in plan.affected_files if file not in submitted]


def render_adaptation_section(plan: MigrationPlan, log: AdaptationLog) -> str:
    """Render the per-file adaptation changes and rationale for the report."""
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
    for entry in sorted(log.entries, key=lambda item: (item.kind, item.source_path)):
        lines.extend(
            (
                f"### `{entry.source_path}` ({entry.kind})",
                "",
                f"- Adapted file: `{entry.output_path}`",
                f"- Why: {entry.rationale}",
                "- What changed:",
                *(
                    [f"  - {change}" for change in entry.changes]
                    or ["  - (no change descriptions were recorded)"]
                ),
            )
        )
        if entry.warnings:
            lines.append("- Validation warnings:")
            lines.extend(f"  - {warning}" for warning in entry.warnings)
        lines.append("")
    gaps = coverage_gaps(plan, log)
    lines.extend(("### Adaptation coverage", ""))
    if gaps:
        lines.extend(
            f"- `{file}`: no adapted version was submitted; its planned changes remain manual."
            for file in gaps
        )
    else:
        lines.append("- Every affected file has a submitted adaptation deliverable.")
    lines.append("")
    return "\n".join(lines)
