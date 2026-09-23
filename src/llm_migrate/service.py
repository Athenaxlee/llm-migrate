"""Shared application service layer used by CLI and MCP transports."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from llm_migrate.adapters.evaluation import BuiltinEvaluationExecutor
from llm_migrate.adapters.evidence import (
    SourceFetcher,
    SourceRefetchRecord,
    refetch_cited_sources,
)
from llm_migrate.adapters.openrouter import (
    ModelsFetcher,
    query_live_pricing,
    query_live_pricing_many,
)
from llm_migrate.analyzers.invocation import (
    analyze_invocation,
    prepare_invocation_migration,
)
from llm_migrate.analyzers.prompt import analyze_prompt
from llm_migrate.core.agent_research import (
    EvidenceReview,
    ExecutionLimits,
    MigrationResearchRequest,
    ModelEndpointIdentity,
    ResearchArtifactValidation,
    ResearchConsensus,
    ResearchScope,
    ResearchTopic,
    SourcePolicy,
    artifact_sha256,
    build_research_consensus,
    topic_for_field_path,
)
from llm_migrate.core.annotations import (
    AnnotatedChange,
    plan_evidence_urls,
    registry_evidence_urls,
)
from llm_migrate.core.artifacts import write_proposal_artifacts
from llm_migrate.core.blockers import (
    DECISIONS_FILENAME,
    RESOLUTION_GUIDANCE,
    BlockerDecisionResult,
    BlockerResolutionSet,
    apply_decisions,
    build_blocker_resolutions,
    downgrade_accepted_prompt_issues,
    load_decision_log,
    render_applied_decision,
    save_decision_log,
    upsert_decision,
)
from llm_migrate.core.change_review import (
    BatchChangeDecisionResult,
    ChangeDecisionRequest,
    ChangeDecisionResult,
    ChangeReviewSet,
    build_change_review,
    reapply_change_decisions,
)
from llm_migrate.core.change_review import (
    record_change_decision as review_record_change_decision,
)
from llm_migrate.core.comparison import compare_models
from llm_migrate.core.consistency import (
    check_cross_surface_consistency,
    render_consistency_section,
)
from llm_migrate.core.evaluation import (
    CustomEvaluator,
    EvaluationExecutor,
    analyze_regressions,
    compare_outputs,
    evaluation_artifact_as_yaml,
    generate_eval_suite,
    manifest_sha256,
    optimize_migration,
    run_migration_eval,
)
from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRunConfig,
    EvaluationSuite,
    MigrationEvalRun,
    OptimizationLimits,
    OptimizationResult,
    OutputComparison,
    RegressionReport,
)
from llm_migrate.core.evaluator_registry import registered_evaluators
from llm_migrate.core.freshness import registry_freshness
from llm_migrate.core.intelligence import check_model_lifecycle, estimate_migration_cost
from llm_migrate.core.invocation_identity import (
    InvocationChoice,
    derive_invocation,
    model_reference_spellings,
    platform_spelling_set,
)
from llm_migrate.core.knowledge import (
    RegistryFreshnessReport,
    RegistryUpdateProposal,
    RegistryValidationReport,
    ResearchResult,
)
from llm_migrate.core.migration import (
    prepare_prompt_migration,
    validate_prompt,
)
from llm_migrate.core.models import (
    ApplicationAnalysis,
    BlockerCategory,
    BlockerDecision,
    ComparisonSeverity,
    InvocationAnalysis,
    InvocationMigrationSpec,
    LivePricingResult,
    MigrationBlocker,
    MigrationCostEstimate,
    MigrationPlan,
    MigrationWorkload,
    ModelComparison,
    ModelDifference,
    ModelLifecycleCheck,
    ModelMatchCandidate,
    ModelMatchResult,
    ModelMatchStatus,
    ModelProfile,
    PricingMatchStatus,
    PromptAnalysis,
    PromptDiscoveryCoverage,
    PromptMigrationSpec,
    PromptSourceConfidence,
    PromptValidationResult,
    RecommendationConstraints,
    RecommendationResult,
    ResolutionKind,
    ResolvedModel,
    UnknownKind,
    ValidationIssue,
    ValidationLevel,
    migration_blocker,
    migration_complexity,
)
from llm_migrate.core.moments import utc_moment
from llm_migrate.core.observations import (
    ObservationResult,
    RunObservation,
    load_observations,
    observation_reasons,
    render_observations_section,
    save_observations,
    upsert_observation,
)
from llm_migrate.core.orchestration import (
    AgentRunner,
    WorkflowOutcome,
    run_agent_research_workflow,
)
from llm_migrate.core.planning import (
    apply_prompt_discovery_dismissals,
    generate_application_migration_plan,
    generate_migration_report,
    migration_manifest_as_yaml,
    out_of_scope_paths,
    unreferenced_prompt_candidates,
)
from llm_migrate.core.probes import render_probe
from llm_migrate.core.prompt_documents import (
    PromptDocumentError,
    extract_prompt_components,
    parse_structured_document,
    source_format,
)
from llm_migrate.core.proposals import propose_registry_update
from llm_migrate.core.recommendation import recommend_models
from llm_migrate.core.registry import ModelRegistry, RegistryError
from llm_migrate.core.research_prompts import ResearchPromptPack, render_research_prompts
from llm_migrate.core.resolver import (
    AmbiguousModelError,
    ModelNotFoundError,
    match_model,
    matching_canonical_names,
    resolve_model,
)
from llm_migrate.core.runstate import atomic_write_text, run_state_lock
from llm_migrate.core.session import (
    SessionRegistryManifest,
    build_session_registry,
    load_session_overlay,
    manifest_summary_lines,
    registry_content_sha256,
)
from llm_migrate.core.snapshot import (
    WorklistSnapshot,
    load_snapshot,
    mark_worklist_requested,
    save_snapshot,
    snapshot_key,
    worklist_requested,
)
from llm_migrate.core.unknowns import close_by_action, with_run_dir
from llm_migrate.core.validation_deliverable import (
    CONTRACT_TEST_RELATIVE_PATH,
    render_contract_test,
)
from llm_migrate.core.workspace import (
    AdaptationLog,
    AdaptationSubmission,
    AdaptationTaskList,
    BatchSubmissionResult,
    FileSubmissionResult,
    GuidanceDisposition,
    MigrationRunConfig,
    MigrationRunFinalization,
    MigrationRunStart,
    PendingSubmission,
    PromptCandidate,
    PromptConsumerConfirmation,
    PromptDiscoveryDismissal,
    PromptDiscoveryUpdate,
    PromptSubmissionResult,
    ResearchNeed,
    RunStatus,
    UnaffectedConfirmation,
    ValidationDisposition,
    apply_submissions,
    consumer_confirmation_map,
    coverage_gaps,
    decision_matches_entry,
    default_run_dir,
    default_run_id,
    derive_adaptation_tasks,
    dismissed_discovery_targets,
    dynamic_prompt_consumers,
    effective_target_id,
    entry_is_unchanged,
    is_consumer_address,
    load_adaptation_log,
    load_change_decision_log,
    load_run_config,
    load_validation_disposition,
    prepare_adapted_file,
    prepare_adapted_prompt,
    read_original_prompt,
    refresh_task_statuses,
    render_adaptation_section,
    run_paths,
    sanitize_run_id,
    save_validation_disposition,
    source_detection_spellings,
    structured_submission_format,
    write_run_config,
)
from llm_migrate.core.workspace import (
    submit_adapted_file as workspace_submit_adapted_file,
)
from llm_migrate.core.workspace import (
    submit_adapted_prompt as workspace_submit_adapted_prompt,
)
from llm_migrate.scanners import scan_application
from llm_migrate.scanners.provenance import resolve_override
from llm_migrate.scanners.python import application_path_resolver, scannable_files

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_SubmissionResultT = TypeVar("_SubmissionResultT", PromptSubmissionResult, FileSubmissionResult)


def _coerced_models(
    model: type[_ModelT],
    items: Sequence[_ModelT | Mapping[str, Any]] | None,
    label: str,
) -> list[_ModelT]:
    """Transport payloads (mappings) coerced into their typed models, fail-closed."""
    coerced: list[_ModelT] = []
    for item in items or []:
        if isinstance(item, model):
            coerced.append(item)
            continue
        try:
            coerced.append(model.model_validate(item))
        except ValidationError as exc:
            raise ValueError(f"invalid {label} {item!r}: {exc}") from exc
    return coerced


def _write_probes(workspace: Path, config: MigrationRunConfig, plan: MigrationPlan) -> list[str]:
    """Write the BYOK probe deliverable of every testable contested unknown."""
    spec = plan.invocation_changes[0] if plan.invocation_changes else None
    written: list[str] = []
    for unknown in plan.unknowns:
        relative = (unknown.data or {}).get("probe_path")
        if spec is None or unknown.kind is not UnknownKind.CONTESTED_EVIDENCE or not relative:
            continue
        content = render_probe(
            spec,
            unknown,
            model_id=effective_target_id(config),
            run_dir=str(workspace.resolve()),
        )
        if content is not None:
            atomic_write_text(workspace / str(relative), content)
            written.append(str(relative))
    return sorted(written)


def _discovery_action(run_dir: str, tasks: AdaptationTaskList, *, strict: bool) -> str:
    """The exact calls that close the discovery_incomplete state.

    A consumer's source is pre-filled only when exactly one prompt file
    matches the keys it reads; with several matches the user must choose.
    """
    parts = [
        f'Prompt discovery is incomplete (coverage {tasks.prompt_coverage}); run_dir="{run_dir}".'
    ]
    if tasks.prompt_candidates:
        listed = ", ".join(f'"{item}"' for item in tasks.prompt_candidates)
        parts.append(
            f"Unreferenced candidate prompt file(s): {listed}. Ask the user which are live "
            "prompts, then call add_prompt_sources(run_dir, paths=[...the live ones...]); "
            "dismiss the rest with add_prompt_sources(run_dir, paths=[], dismiss=[...], "
            'rationale="<the user\'s reason>").'
        )
    if strict and tasks.dynamic_prompt_consumers:
        calls = []
        for item in tasks.dynamic_prompt_consumers:
            reads = f" reads {', '.join(item.access_keys)}" if item.access_keys else ""
            if len(item.matching_sources) == 1:
                source = f'"{item.matching_sources[0]}"'
            elif item.matching_sources:
                source = f"<ask the user: one of {', '.join(item.matching_sources)}>"
            else:
                source = "<the prompt file it reads>"
            calls.append(
                f"{item.location} ({item.keyword}{reads}): confirm_prompt_consumer(run_dir, "
                f'location="{item.location}", source_path={source})'
            )
        parts.append(
            "STRICT MODE: every dynamic prompt consumer must be confirmed or dismissed — "
            + "; ".join(calls)
            + ". Dismiss genuinely runtime-built consumers with add_prompt_sources(run_dir, "
            'paths=[], dismiss=["<path:line>", ...], rationale="<the user\'s reason>"). '
            "Ask the user; never decide for them."
        )
    return " ".join(parts)


def _action_required_lines(
    plan: MigrationPlan,
    *,
    gaps: list[str],
    unaffected: list[str],
    undecided: list[str],
    consistency: list[str],
    validation_disposition: ValidationDisposition | None,
    validation_problem: str | None,
) -> list[str]:
    """What a human must still decide or do, in the order the workflow needs it."""
    lines: list[str] = []
    if plan.blockers:
        lines.append(
            f"**{len(plan.blockers)} unresolved blocker(s)** — each needs your decision "
            "(get_blocker_resolutions → record_blocker_decision)."
        )
    stale = [item for item in plan.decisions if item.status == "stale"]
    if stale:
        lines.append(f"**{len(stale)} STALE blocker decision(s)** — no longer applied; review.")
    if plan.prompt_discovery.coverage is not PromptDiscoveryCoverage.RESOLVED:
        candidates = unreferenced_prompt_candidates(plan)
        lines.append(
            f"**Prompt coverage is {plan.prompt_discovery.coverage.value}** — "
            + (
                f"{len(candidates)} candidate prompt file(s) were not prepared; include "
                "the real ones with add_prompt_sources, or dismiss them with the "
                "user's rationale (see Unresolved unknowns)."
                if candidates
                else "some prompt consumers have no static source; confirm each with "
                "confirm_prompt_consumer or dismiss it with the user's rationale."
            )
        )
    still_open = [item for item in plan.unknowns if item.is_open]
    if still_open:
        lines.append(
            f"**{len(still_open)} open unknown(s)** — each lists why it matters, its exact "
            "action, and what closes it under Unresolved unknowns."
        )
    if gaps:
        lines.append(f"**{len(gaps)} affected file(s) lack a deliverable:** " + ", ".join(gaps))
    if unaffected:
        lines.append(f"**{len(unaffected)} unaffected file(s) await one confirm_unaffected call.**")
    if undecided:
        lines.append(
            f"**{len(undecided)} deliverable(s) have changes awaiting your review "
            "decision** (get_change_review)."
        )
    if consistency:
        lines.append(
            f"**{len(consistency)} cross-surface consistency finding(s)** — see "
            "Cross-surface consistency below."
        )
    if validation_disposition is None:
        lines.append(
            "**Validation not recorded** — run the generated contract test or a BYOK "
            "evaluation, record the outcome with record_validation_disposition, then "
            "finalize again (validation is a two-pass flow by design)."
        )
    elif validation_problem is not None:
        lines.append(f"**NOT VALIDATED** — {validation_problem}.")
    return lines


def _with_action_required(report: str, lines: list[str]) -> str:
    """Insert the "Action required" section directly under the report title."""
    section = "\n".join(
        [
            "## Action required",
            "",
            *([f"- {line}" for line in lines] or ["- Nothing requires a human decision."]),
            "",
        ]
    )
    marker = "\n## Summary"
    index = report.find(marker)
    if index == -1:
        return section + "\n" + report
    return report[: index + 1] + section + "\n" + report[index + 1 :]


def _validation_problem(
    disposition: ValidationDisposition | None, current_manifest_sha256: str
) -> str | None:
    """Why a recorded disposition does not evidence validation, if it does not."""
    if disposition is None:
        return None
    if disposition.method == "generated_tests":
        if disposition.outcome is None:
            return (
                "outcome not recorded (recorded before v1.5.2); re-record it with the "
                "test run's outcome and summary line"
            )
        if disposition.outcome != "run_passed":
            return f"the generated contract test outcome is {disposition.outcome}, not a pass"
    if disposition.method == "byok_evaluation":
        if disposition.bound_manifest_sha256 is None:
            return (
                "outcome not recorded (recorded before v1.5.2 without an evaluation run "
                "artifact); re-record it with evaluation_run_path"
            )
        if disposition.bound_manifest_sha256 != current_manifest_sha256:
            return (
                "the recorded evaluation is bound to a superseded plan (a later decision "
                "or submission changed it); regenerate the suite, re-run the evaluation, "
                "and re-record the disposition"
            )
    return None


def _render_validation_section(
    disposition: ValidationDisposition | None,
    contract_test_path: str | None,
    problem: str | None = None,
) -> str:
    """The report section recording how the migration was (or was not) validated."""
    lines = ["## Validation", ""]
    if disposition is None:
        lines.append(
            "- No validation disposition is recorded yet. Validate with a BYOK "
            "evaluation run (generate_eval_suite / run_migration_eval), run the "
            "generated contract test, or record an explicit accept with "
            "record_validation_disposition."
        )
    else:
        labels = {
            "byok_evaluation": "BYOK evaluation run",
            "generated_tests": "user-executed generated contract tests",
            "accepted_without_validation": "ACCEPTED WITHOUT VALIDATION",
        }
        line = (
            f"- Validated via {labels[disposition.method]} ({disposition.decided_on.isoformat()})."
        )
        if disposition.outcome_summary:
            line += f" Outcome ({disposition.outcome}): {disposition.outcome_summary}"
        if disposition.evaluation_run_path:
            line += f" Evaluation run: `{disposition.evaluation_run_path}`."
        if disposition.rationale:
            line += f" Rationale: {disposition.rationale}"
        lines.append(line)
        if problem is not None:
            lines.append(f"- **NOT VALIDATED:** {problem}.")
    if contract_test_path is not None:
        lines.append(
            f"- Generated contract test: `{contract_test_path}` — a mocked "
            "request-shape test for the target invocation; wire build_request() to "
            "the adapted application and run it yourself (the toolkit never executes "
            "application code)."
        )
    lines.append("")
    return "\n".join(lines)


class MigrationService:
    """Thin transport-independent entry point for deterministic operations."""

    def __init__(self, registry: ModelRegistry, registry_root: Path | None = None) -> None:
        self.registry = registry
        self.registry_root = registry_root

    @classmethod
    def from_directory(cls, directory: Path | str) -> MigrationService:
        path = Path(directory)
        if (path / "models").is_dir():
            return cls(ModelRegistry.from_root(path), registry_root=path)
        return cls(ModelRegistry.from_directory(path), registry_root=path)

    def validate_registry(self) -> RegistryValidationReport:
        return self.registry.validation_report()

    def check_registry_freshness(self, as_of_date: date) -> RegistryFreshnessReport:
        return registry_freshness(self.registry.all(), as_of_date)

    def propose_registry_update(self, research: ResearchResult) -> RegistryUpdateProposal:
        return propose_registry_update(research, self.registry)

    def write_registry_proposal(
        self,
        research: ResearchResult,
        output_directory: Path,
    ) -> tuple[RegistryUpdateProposal, Path]:
        proposal = self.propose_registry_update(research)
        target = write_proposal_artifacts(proposal, research, output_directory)
        return proposal, target

    def resolve_model(
        self,
        identifier: str,
        platform: str | None = None,
        endpoint: str | None = None,
    ) -> ResolvedModel:
        return resolve_model(self.registry, identifier, platform, endpoint)

    def match_model(
        self,
        identifier: str,
        platform: str | None = None,
        endpoint: str | None = None,
    ) -> ModelMatchResult:
        """Registry-first lenient matching; suggests candidates instead of failing."""
        return match_model(self.registry, identifier, platform, endpoint)

    def get_model_profile(self, identifier: str) -> ModelProfile:
        return self.resolve_model(identifier).profile

    def check_model_lifecycle(
        self,
        identifier: str,
        *,
        as_of_date: date | None = None,
    ) -> ModelLifecycleCheck:
        return check_model_lifecycle(
            self.get_model_profile(identifier),
            as_of_date=as_of_date or date.today(),
        )

    def estimate_migration_cost(
        self,
        source: str,
        target: str,
        workload: MigrationWorkload,
    ) -> MigrationCostEstimate:
        return estimate_migration_cost(
            self.get_model_profile(source),
            self.get_model_profile(target),
            workload,
        )

    def compare_models(
        self,
        source: str,
        target: str,
        source_platform: str | None = None,
        target_platform: str | None = None,
        source_endpoint: str | None = None,
        target_endpoint: str | None = None,
        include_live_pricing: bool = False,
        pricing_timeout: float = 5.0,
        pricing_fetcher: ModelsFetcher | None = None,
    ) -> ModelComparison:
        source_resolution = self.resolve_model(source, source_platform, source_endpoint)
        target_resolution = self.resolve_model(target, target_platform, target_endpoint)
        source_profile = source_resolution.profile
        target_profile = target_resolution.profile
        comparison = compare_models(
            source_profile,
            target_profile,
            source_resolution.platform,
            target_resolution.platform,
        )
        knowledge_differences = [
            ModelDifference(
                category="migration_knowledge",
                source_value=change.field_path,
                target_value=change.statement,
                severity=change.severity,
                migration_impact=change.statement,
                recommended_action=change.recommended_action,
                supporting_sources=change.supporting_sources,
                knowledge_id=knowledge.id,
                target_evidence_url=next(
                    (
                        str(source.url)
                        for source in knowledge.sources
                        if source.id in change.supporting_sources and source.url is not None
                    ),
                    None,
                ),
            )
            for knowledge in self.registry.migrations_for(
                source_profile.identity.canonical_name,
                target_profile.identity.canonical_name,
            )
            for change in knowledge.changes
        ]
        if knowledge_differences:
            all_differences = [*comparison.differences, *knowledge_differences]
            severity_order = {severity: index for index, severity in enumerate(ComparisonSeverity)}
            highest = max(
                (item.severity for item in all_differences),
                key=lambda severity: severity_order[severity],
            )
            comparison = comparison.model_copy(
                update={"differences": all_differences, "highest_severity": highest}
            )
        if include_live_pricing:
            overlays = query_live_pricing_many(
                [source_profile, target_profile],
                timeout=pricing_timeout,
                fetcher=pricing_fetcher,
            )
            comparison = comparison.model_copy(update={"pricing_overlays": overlays})
        return comparison

    def recommend_models(
        self,
        constraints: RecommendationConstraints,
        application_analysis: ApplicationAnalysis | None = None,
        as_of_date: date | None = None,
        include_live_pricing: bool = False,
        pricing_timeout: float = 5.0,
        pricing_fetcher: ModelsFetcher | None = None,
    ) -> RecommendationResult:
        if constraints.source_model:
            source = self.resolve_model(constraints.source_model)
            constraints = constraints.model_copy(
                update={"source_model": source.identity.canonical_name}
            )
        elif application_analysis and len(application_analysis.requirements.source_models) == 1:
            detected_source = next(iter(application_analysis.requirements.source_models))
            try:
                source = self.resolve_model(detected_source)
            except RegistryError:
                pass
            else:
                constraints = constraints.model_copy(
                    update={"source_model": source.identity.canonical_name}
                )
        if (
            constraints.source_platform is None
            and application_analysis
            and len(application_analysis.requirements.source_platforms) == 1
        ):
            constraints = constraints.model_copy(
                update={
                    "source_platform": next(
                        iter(application_analysis.requirements.source_platforms)
                    )
                }
            )
        result = recommend_models(
            self.registry,
            constraints,
            requirements=application_analysis.requirements if application_analysis else None,
            as_of_date=as_of_date,
        )
        if include_live_pricing:
            profiles = [self.registry.get(item.canonical_name) for item in result.recommendations]
            overlays = [
                item
                for item in query_live_pricing_many(
                    profiles,
                    timeout=pricing_timeout,
                    fetcher=pricing_fetcher,
                )
                if item.status is not PricingMatchStatus.UNMAPPED
            ]
            result = result.model_copy(update={"pricing_overlays": overlays})
        return result

    def _model_id_spellings(self) -> list[str]:
        """Every reviewed model-id spelling in the registry (bare + selectors).

        Anchors semantic config-coupling detection: a config document that
        names any of these spellings is model-coupled.
        """
        spellings: set[str] = set()
        for profile in self.registry.all():
            for platform in profile.platforms:
                spellings.add(platform.model_id)
                if platform.invocation is not None:
                    spellings.update(
                        selector.model_id for selector in platform.invocation.selectors
                    )
        return sorted(spellings)

    def scan_application(
        self,
        root: Path | str,
        *,
        prompt_sources: Sequence[str] | None = None,
        consumer_confirmations: Mapping[str, str] | None = None,
        dismissed_consumers: Sequence[str] | None = None,
    ) -> ApplicationAnalysis:
        return scan_application(
            root,
            prompt_sources=prompt_sources,
            known_model_ids=self._model_id_spellings(),
            consumer_confirmations=consumer_confirmations,
            dismissed_consumers=dismissed_consumers,
        )

    def _scan_for_run(
        self, config: MigrationRunConfig, service: MigrationService | None = None
    ) -> ApplicationAnalysis:
        """Scan a run's application with every recorded discovery decision."""
        dismissed = dismissed_discovery_targets(config)
        return (service or self).scan_application(
            config.application_root,
            prompt_sources=config.prompt_sources or None,
            consumer_confirmations=consumer_confirmation_map(config) or None,
            dismissed_consumers=[item for item in dismissed if is_consumer_address(item)] or None,
        )

    def query_live_pricing(
        self,
        identifier: str,
        *,
        timeout: float = 5.0,
        fetcher: ModelsFetcher | None = None,
    ) -> LivePricingResult:
        return query_live_pricing(
            self.resolve_model(identifier).profile,
            timeout=timeout,
            fetcher=fetcher,
        )

    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        return analyze_prompt(prompt)

    def prepare_prompt_migration(
        self,
        source: str,
        target: str,
        prompt: str,
        *,
        source_path: str | None = None,
        source_role: Literal["system", "user", "developer", "unknown"] = "unknown",
        source_platform: str | None = None,
        target_platform: str | None = None,
    ) -> PromptMigrationSpec:
        return prepare_prompt_migration(
            self.resolve_model(source, source_platform),
            self.resolve_model(target, target_platform),
            prompt,
            source_path=source_path,
            source_role=source_role,
        )

    def validate_prompt(
        self,
        target: str,
        prompt: str,
        *,
        source_path: str | None = None,
        target_platform: str | None = None,
        target_endpoint: str | None = None,
    ) -> PromptValidationResult:
        resolution = self.resolve_model(target)
        ambiguous_platform = False
        if target_platform:
            try:
                resolution = self.resolve_model(target, target_platform, target_endpoint)
            except ModelNotFoundError:
                pass
            except AmbiguousModelError:
                ambiguous_platform = True
        result = validate_prompt(
            resolution,
            prompt,
            source_path=source_path,
            target_platform=target_platform,
        )
        if ambiguous_platform:
            issue = ValidationIssue(
                code="ambiguous_target_platform",
                level=ValidationLevel.BLOCKER,
                message=(
                    f"Target model has multiple {target_platform} representations; "
                    "endpoint context is required."
                ),
                source_path=source_path,
            )
            result = result.model_copy(update={"valid": False, "issues": [*result.issues, issue]})
        return result

    def analyze_invocation(
        self,
        application: Path | str | ApplicationAnalysis,
        *,
        target: str | None = None,
        target_platform: str | None = None,
    ) -> InvocationAnalysis:
        analysis = (
            application
            if isinstance(application, ApplicationAnalysis)
            else self.scan_application(application)
        )
        resolution = self.resolve_model(target, target_platform) if target is not None else None
        return analyze_invocation(analysis, resolution)

    def prepare_invocation_migration(
        self,
        application: Path | str | ApplicationAnalysis,
        source: str,
        target: str,
        *,
        source_platform: str | None = None,
        target_platform: str | None = None,
        source_endpoint: str | None = None,
        target_endpoint: str | None = None,
    ) -> InvocationMigrationSpec:
        analysis = (
            application
            if isinstance(application, ApplicationAnalysis)
            else self.scan_application(application)
        )
        source_resolution = self.resolve_model(source, source_platform, source_endpoint)
        consistency_warnings: list[str] = []
        consistency_blockers: list[MigrationBlocker] = []
        detected_canonical: set[str] = set()
        for identifier in sorted(analysis.requirements.source_models):
            try:
                detected_canonical.add(self.resolve_model(identifier).canonical_name)
            except AmbiguousModelError:
                # Same model id on several platform representations still names
                # exactly one canonical model for consistency purposes.
                names = matching_canonical_names(self.registry, identifier)
                if len(names) == 1:
                    detected_canonical.add(next(iter(names)))
                else:
                    consistency_warnings.append(
                        f"Detected model identifier {identifier!r} is ambiguous; "
                        "matches: " + ", ".join(sorted(names)) + "."
                    )
            except ModelNotFoundError:
                consistency_warnings.append(
                    f"Detected model identifier {identifier!r} is not in the registry."
                )
        if detected_canonical and source_resolution.canonical_name not in detected_canonical:
            consistency_blockers.append(
                migration_blocker(
                    code="source_model_mismatch",
                    category=BlockerCategory.SOURCE_CONSISTENCY,
                    message=(
                        "Declared source model does not match detected model(s): "
                        + ", ".join(sorted(detected_canonical))
                        + "."
                    ),
                    data={
                        "declared": source_resolution.canonical_name,
                        "detected": sorted(detected_canonical),
                        "detected_platforms": sorted(analysis.requirements.source_platforms),
                    },
                )
            )
        elif len(detected_canonical) > 1:
            consistency_warnings.append(
                "Multiple source models were detected; this preparation covers only "
                f"{source_resolution.canonical_name}."
            )
        detected_providers = analysis.requirements.source_providers
        if detected_providers and source_resolution.identity.provider not in detected_providers:
            consistency_blockers.append(
                migration_blocker(
                    code="source_provider_mismatch",
                    category=BlockerCategory.SOURCE_CONSISTENCY,
                    message=(
                        "Declared source provider does not match detected provider(s): "
                        + ", ".join(sorted(detected_providers))
                        + "."
                    ),
                    data={
                        "declared": source_resolution.identity.provider,
                        "detected": sorted(detected_providers),
                        "detected_models": sorted(detected_canonical),
                    },
                )
            )
        elif len(detected_providers) > 1:
            consistency_warnings.append(
                "Multiple source providers were detected; review each invocation separately."
            )
        if (
            source_resolution.platform
            and analysis.requirements.source_platforms
            and source_resolution.platform.platform not in analysis.requirements.source_platforms
        ):
            consistency_blockers.append(
                migration_blocker(
                    code="source_platform_mismatch",
                    category=BlockerCategory.SOURCE_CONSISTENCY,
                    message=(
                        "Declared source platform does not match detected platform(s): "
                        + ", ".join(sorted(analysis.requirements.source_platforms))
                        + "."
                    ),
                    data={
                        "declared": source_resolution.platform.platform,
                        "detected": sorted(analysis.requirements.source_platforms),
                    },
                )
            )
        elif len(analysis.requirements.source_platforms) > 1:
            consistency_warnings.append(
                "Multiple source platforms were detected; review each invocation separately."
            )
        return prepare_invocation_migration(
            analysis,
            source_resolution,
            self.resolve_model(target, target_platform, target_endpoint),
            preparation_warnings=consistency_warnings,
            preparation_blockers=consistency_blockers,
        )

    def generate_migration_plan(
        self,
        application: Path | str | ApplicationAnalysis,
        source: str,
        target: str,
        *,
        source_platform: str | None = None,
        target_platform: str | None = None,
        source_endpoint: str | None = None,
        target_endpoint: str | None = None,
        prompt_sources: Sequence[str] | None = None,
    ) -> MigrationPlan:
        analysis = (
            application
            if isinstance(application, ApplicationAnalysis)
            else self.scan_application(application, prompt_sources=prompt_sources)
        )
        source_resolution = self.resolve_model(source, source_platform, source_endpoint)
        target_resolution = self.resolve_model(target, target_platform, target_endpoint)
        if source_resolution.platform is None:
            if len(source_resolution.platforms) != 1:
                raise ValueError("Source platform is required for migration planning.")
            source_resolution = self.resolve_model(source, source_resolution.platforms[0].platform)
        if target_resolution.platform is None:
            if len(target_resolution.platforms) != 1:
                raise ValueError("Target platform is required for migration planning.")
            target_resolution = self.resolve_model(target, target_resolution.platforms[0].platform)
        source_representation = source_resolution.platform
        target_representation = target_resolution.platform
        if source_representation is None or target_representation is None:
            raise ValueError("Source and target platform representations are required.")
        comparison = self.compare_models(
            source_resolution.canonical_name,
            target_resolution.canonical_name,
            source_representation.platform,
            target_representation.platform,
            source_endpoint or source_representation.endpoint,
            target_endpoint or target_representation.endpoint,
        )
        invocation = self.prepare_invocation_migration(
            analysis,
            source_resolution.canonical_name,
            target_resolution.canonical_name,
            source_platform=source_representation.platform,
            target_platform=target_representation.platform,
            source_endpoint=source_endpoint or source_representation.endpoint,
            target_endpoint=target_endpoint or target_representation.endpoint,
        )
        return generate_application_migration_plan(
            analysis,
            source_resolution,
            target_resolution,
            comparison,
            invocation,
        )

    def migration_manifest_as_yaml(self, plan: MigrationPlan) -> str:
        return migration_manifest_as_yaml(plan)

    def migration_report(self, plan: MigrationPlan) -> str:
        return generate_migration_report(plan)

    def generate_migration_report(
        self,
        application: Path | str | ApplicationAnalysis,
        source: str,
        target: str,
        *,
        source_platform: str | None = None,
        target_platform: str | None = None,
        source_endpoint: str | None = None,
        target_endpoint: str | None = None,
        prompt_sources: Sequence[str] | None = None,
    ) -> str:
        plan = self.generate_migration_plan(
            application,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            prompt_sources=prompt_sources,
        )
        return self.migration_report(plan)

    def generate_eval_suite(
        self,
        plan: MigrationPlan,
        cases: list[EvaluationCase],
        *,
        name: str = "migration-evaluation",
    ) -> EvaluationSuite:
        return generate_eval_suite(plan, cases, name=name)

    def run_migration_eval(
        self,
        suite: EvaluationSuite,
        source_config: EvaluationRunConfig,
        target_config: EvaluationRunConfig,
        *,
        executor: EvaluationExecutor | None = None,
        evaluators: Mapping[str, CustomEvaluator] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> MigrationEvalRun:
        return run_migration_eval(
            suite,
            source_config,
            target_config,
            executor or BuiltinEvaluationExecutor(),
            evaluators={**registered_evaluators(), **(evaluators or {})},
            clock=clock,
        )

    def compare_outputs(
        self,
        source: EvaluationCaseResult,
        target: EvaluationCaseResult,
    ) -> OutputComparison:
        return compare_outputs(source, target)

    def analyze_regressions(self, run: MigrationEvalRun) -> RegressionReport:
        return analyze_regressions(run)

    def optimize_migration(
        self,
        report: RegressionReport,
        candidate_runs: Mapping[str, MigrationEvalRun] | None = None,
        *,
        limits: OptimizationLimits | None = None,
    ) -> OptimizationResult:
        return optimize_migration(report, candidate_runs, limits=limits)

    def evaluation_artifact_as_yaml(
        self,
        value: EvaluationSuite | MigrationEvalRun | RegressionReport | OptimizationResult,
    ) -> str:
        return evaluation_artifact_as_yaml(value)

    # ------------------------------------------------------------------
    # V1.1 on-demand agent-researched migration stage operations
    # ------------------------------------------------------------------

    _DEFAULT_MODEL_TOPICS = (
        ResearchTopic.PRICING,
        ResearchTopic.LIFECYCLE,
        ResearchTopic.AVAILABILITY,
        ResearchTopic.CAPABILITIES,
        ResearchTopic.PARAMETERS,
        ResearchTopic.PROMPTING_GUIDANCE,
    )

    def _side_research_needs(
        self,
        identity: ModelEndpointIdentity,
        as_of: date,
    ) -> tuple[str | None, list[ResearchTopic]]:
        """Return (canonical name if resolvable, topics still needing research)."""
        try:
            resolved = self.resolve_model(identity.model, identity.platform, identity.endpoint)
        except RegistryError:
            try:
                resolved = self.resolve_model(identity.model)
            except RegistryError:
                return None, list(self._DEFAULT_MODEL_TOPICS)
        profile = self.registry.get(resolved.identity.canonical_name)
        report = registry_freshness([profile], as_of)
        stale = sorted({section.category.value for section in report.sections if section.stale})
        return resolved.identity.canonical_name, [ResearchTopic(value) for value in stale]

    def create_migration_research_request(
        self,
        application: Path | str | ApplicationAnalysis,
        *,
        run_id: str,
        source: ModelEndpointIdentity,
        target: ModelEndpointIdentity,
        as_of: date,
        topics: list[ResearchTopic] | None = None,
        source_policy: SourcePolicy | None = None,
        limits: ExecutionLimits | None = None,
    ) -> MigrationResearchRequest:
        """Scan locally, then bound research to what is actually missing or stale."""
        analysis = (
            application
            if isinstance(application, ApplicationAnalysis)
            else self.scan_application(application)
        )
        requirements_sha256 = artifact_sha256(analysis.requirements)

        source_name, source_topics = self._side_research_needs(source, as_of)
        target_name, target_topics = self._side_research_needs(target, as_of)
        scopes: list[ResearchScope] = []
        needed_topics: set[ResearchTopic] = set()
        if source_name is None or source_topics:
            scopes.append(ResearchScope.SOURCE)
            needed_topics.update(source_topics or self._DEFAULT_MODEL_TOPICS)
        if target_name is None or target_topics:
            scopes.append(ResearchScope.TARGET)
            needed_topics.update(target_topics or self._DEFAULT_MODEL_TOPICS)
        if (
            source_name is None
            or target_name is None
            or not self.registry.migrations_for(source_name, target_name)
        ):
            scopes.append(ResearchScope.PAIR)
            needed_topics.add(ResearchTopic.MIGRATION_BEHAVIOR)
        if not scopes:
            raise RegistryError(
                "the canonical registry already covers this migration with fresh "
                "knowledge; explicit research is unnecessary"
            )
        selected_topics = topics if topics is not None else sorted(needed_topics)
        return MigrationResearchRequest(
            run_id=run_id,
            source=source,
            target=target,
            application_requirements_sha256=requirements_sha256,
            topics=list(selected_topics),
            scopes=scopes,
            as_of=as_of,
            source_policy=source_policy or SourcePolicy(),
            limits=limits or ExecutionLimits(),
        )

    def validate_research_result(
        self,
        research: ResearchResult,
        request: MigrationResearchRequest,
    ) -> list[str]:
        """Deterministic scope/policy gate; returns human-actionable problems."""
        problems: list[str] = []
        expected_subjects = {
            request.source.model,
            request.target.model,
            f"{request.source.model} -> {request.target.model}",
        }
        if research.subject not in expected_subjects:
            problems.append(
                f"research subject {research.subject!r} is not one of the requested "
                f"identities: {sorted(expected_subjects)}"
            )
        if research.research_date > request.as_of:
            problems.append(
                f"research date {research.research_date.isoformat()} is after the "
                f"requested as-of date {request.as_of.isoformat()}"
            )
        requested = set(request.topics)
        for claim in research.claims:
            topic = topic_for_field_path(claim.field_path)
            if topic is not None and topic not in requested:
                problems.append(f"claim {claim.id!r} researches unrequested topic {topic.value!r}")
        for source in research.sources:
            supports_claim = any(source.id in claim.sources for claim in research.claims)
            supports_observation = any(
                source.id in observation.sources for observation in research.observations
            )
            if not supports_claim and not supports_observation:
                problems.append(f"source {source.id!r} supports no claim or observation")
        return problems

    def validate_evidence_review(
        self,
        review: EvidenceReview,
        research: ResearchResult,
    ) -> list[str]:
        """Deterministic review-integrity gate; returns human-actionable problems."""
        problems: list[str] = []
        expected = artifact_sha256(research)
        if review.research_sha256 != expected:
            problems.append(
                f"review does not reference this research artifact (expected {expected})"
            )
        claim_ids = {claim.id for claim in research.claims}
        verdict_ids = {verdict.claim_id for verdict in review.verdicts}
        for missing in sorted(claim_ids - verdict_ids):
            problems.append(f"claim {missing!r} has no review verdict")
        for unknown in sorted(verdict_ids - claim_ids):
            problems.append(f"verdict references unknown claim {unknown!r}")
        return problems

    def validate_research_artifact(
        self,
        run_dir: Path | str,
        scope: str,
    ) -> ResearchArtifactValidation:
        """Validate one scope's research artifacts directly from the run workspace.

        Reads `<run_dir>/request.yaml`, the researcher artifact at
        `research/<scope>.yaml`, and — when it already exists — the reviewer
        artifact at `review/<scope>.yaml`, then runs the same deterministic
        gates as `validate_research_result` / `validate_evidence_review`.
        Nothing has to be resent through the transport payload.
        """
        workspace = Path(run_dir)
        request_path = workspace / "request.yaml"
        if not request_path.is_file():
            raise ValueError(
                f"{request_path} does not exist; create the bounded research request first"
            )
        try:
            request = MigrationResearchRequest.model_validate(
                yaml.safe_load(request_path.read_text(encoding="utf-8"))
            )
        except (yaml.YAMLError, ValidationError) as exc:
            raise ValueError(f"invalid research request {request_path}: {exc}") from exc
        try:
            scope_value = ResearchScope(scope)
        except ValueError:
            valid_scopes = ", ".join(item.value for item in ResearchScope)
            return ResearchArtifactValidation(
                run_id=request.run_id,
                scope=scope,
                valid=False,
                problems=[f"unknown scope {scope!r}; valid scopes: {valid_scopes}"],
                message=f"Invalid: unknown scope {scope!r}.",
            )
        research_path = workspace / "research" / f"{scope_value.value}.yaml"
        review_path = workspace / "review" / f"{scope_value.value}.yaml"
        result = ResearchArtifactValidation(
            run_id=request.run_id,
            scope=scope_value.value,
            research_path=str(research_path),
            review_path=str(review_path),
            valid=False,
            message="",
        )
        problems: list[str] = []
        if scope_value not in request.scopes:
            problems.append(
                f"scope {scope_value.value!r} is not part of this run's research request; "
                "requested scopes: " + ", ".join(item.value for item in request.scopes)
            )
        research: ResearchResult | None = None
        if not research_path.is_file():
            problems.append(
                f"research artifact {research_path} does not exist; write the "
                "researcher's YAML output there first"
            )
        else:
            try:
                research = ResearchResult.model_validate(
                    yaml.safe_load(research_path.read_text(encoding="utf-8"))
                )
            except (yaml.YAMLError, ValidationError) as exc:
                problems.append(f"research artifact {research_path} is invalid: {exc}")
        review_status = "review artifact not written yet (validate it here once it is)"
        review: EvidenceReview | None = None
        if research is not None:
            problems.extend(self.validate_research_result(research, request))
            result = result.model_copy(update={"research_checked": True})
            if review_path.is_file():
                try:
                    review = EvidenceReview.model_validate(
                        yaml.safe_load(review_path.read_text(encoding="utf-8"))
                    )
                except (yaml.YAMLError, ValidationError) as exc:
                    problems.append(f"review artifact {review_path} is invalid: {exc}")
                else:
                    problems.extend(self.validate_evidence_review(review, research))
                    result = result.model_copy(update={"review_checked": True})
                review_status = "review artifact checked"
        valid = not problems
        checked = "research artifact checked" if result.research_checked else "research not checked"
        return result.model_copy(
            update={
                "valid": valid,
                "problems": problems,
                "message": (
                    ("Valid" if valid else "Invalid")
                    + f" ({checked}; {review_status})."
                    + ("" if valid else " Fix only the reported problems and re-validate.")
                ),
            }
        )

    def build_research_consensus(
        self,
        research: ResearchResult,
        review: EvidenceReview,
        request: MigrationResearchRequest,
    ) -> ResearchConsensus:
        return build_research_consensus(research, review, request.topics, request.source_policy)

    def run_agent_research(
        self,
        request: MigrationResearchRequest,
        runner: AgentRunner,
        workspace: Path,
        *,
        now: datetime,
        overlay_ttl: timedelta | None = None,
        shadow_canonical: list[str] | None = None,
    ) -> WorkflowOutcome:
        """Drive the full research workflow with a host-supplied `AgentRunner`."""
        if self.registry_root is None:
            raise RegistryError("agent research requires a directory-backed canonical registry")
        kwargs: dict[str, Any] = {}
        if overlay_ttl is not None:
            kwargs["overlay_ttl"] = overlay_ttl
        return run_agent_research_workflow(
            self.registry,
            request,
            runner,
            workspace,
            now=now,
            base_registry_sha256=registry_content_sha256(self.registry_root),
            shadow_canonical=shadow_canonical,
            **kwargs,
        )

    def refetch_research_sources(
        self,
        research: ResearchResult,
        *,
        timeout: float = 5.0,
        fetcher: SourceFetcher | None = None,
        retrieved_at: datetime | None = None,
    ) -> list[SourceRefetchRecord]:
        """Explicitly refetch cited sources; reachability evidence only."""
        return refetch_cited_sources(
            research,
            timeout=timeout,
            fetcher=fetcher,
            retrieved_at=retrieved_at,
        )

    def load_session_service(
        self,
        run_dir: Path | str,
        *,
        as_of: datetime,
    ) -> tuple[MigrationService, SessionRegistryManifest]:
        """Build a service over the canonical registry plus one run's overlay.

        The overlay is hash-verified, must be unexpired and
        `session_agent_reviewed`, and never mutates checked-in registry files.
        `as_of` is normalized to aware UTC (naive means UTC), so host-supplied
        timestamps never hit an aware/naive comparison error.
        """
        as_of = utc_moment(as_of)
        manifest, candidates, knowledge = load_session_overlay(Path(run_dir))
        if self.registry_root is not None:
            current = registry_content_sha256(self.registry_root)
            if current != manifest.base_registry_sha256:
                raise RegistryError(
                    "the canonical registry changed since this session overlay was "
                    "built; re-run research or rebuild the overlay"
                )
        merged = build_session_registry(
            self.registry,
            manifest,
            candidates,
            knowledge,
            as_of=as_of,
        )
        return MigrationService(merged, registry_root=self.registry_root), manifest

    # ------------------------------------------------------------------
    # V1.2 guided migration run workspace
    # ------------------------------------------------------------------

    def _platform_selected(self, match: ModelMatchResult, side: str) -> ModelMatchResult:
        """Pin a resolved match to exactly one platform representation."""
        if match.status is not ModelMatchStatus.RESOLVED or match.resolution is None:
            return match
        resolution = match.resolution
        if resolution.platform is not None:
            return match
        platforms = resolution.profile.platforms
        if len(platforms) == 1:
            pinned = self.resolve_model(resolution.canonical_name, platforms[0].platform)
            return match.model_copy(
                update={
                    "resolution": pinned,
                    "platform": platforms[0].platform,
                    "model_id": platforms[0].model_id,
                    "notes": [
                        *match.notes,
                        f"Used {platforms[0].platform!r}, the only platform representation.",
                    ],
                }
            )
        identity = resolution.profile.identity
        return ModelMatchResult(
            query=match.query,
            platform_query=match.platform_query,
            status=ModelMatchStatus.NEEDS_CONFIRMATION,
            candidates=[
                ModelMatchCandidate(
                    canonical_name=identity.canonical_name,
                    display_name=identity.display_name,
                    provider=identity.provider,
                    matched_identifier=item.model_id,
                    platform=item.platform,
                    similarity=1.0,
                    reason="platform selection required",
                )
                for item in platforms
            ],
            notes=[
                *match.notes,
                f"{identity.canonical_name!r} has multiple platform representations; "
                f"the {side} platform must be selected.",
            ],
            guidance=(
                f"Ask the user which platform the {side} model runs on, then retry with "
                f"the {side}_platform argument set."
            ),
        )

    def _research_need(
        self,
        application: ApplicationAnalysis,
        run_dir: Path,
        run_id: str,
        source: ModelEndpointIdentity,
        target: ModelEndpointIdentity,
        as_of: date,
        write_request: bool,
    ) -> ResearchNeed:
        """Decide whether bounded research would improve this run, and record why."""
        try:
            request = self.create_migration_research_request(
                application,
                run_id=run_id,
                source=source,
                target=target,
                as_of=as_of,
            )
        except RegistryError:
            return ResearchNeed(
                level="none",
                reasons=[
                    "The canonical registry already covers this migration with fresh knowledge."
                ],
            )
        reasons: list[str] = []
        side_names: dict[str, str | None] = {}
        for label, identity in (("source", source), ("target", target)):
            name, stale = self._side_research_needs(identity, as_of)
            side_names[label] = name
            if name is None:
                reasons.append(f"The {label} model {identity.model!r} is not in the registry.")
            elif stale:
                reasons.append(
                    f"Canonical facts for the {label} model {name!r} are stale for: "
                    + ", ".join(topic.value for topic in stale)
                    + "."
                )
        source_name = side_names["source"]
        target_name = side_names["target"]
        if not (
            source_name and target_name and self.registry.migrations_for(source_name, target_name)
        ):
            reasons.append(
                "No reviewed pair-specific migration knowledge exists for this exact "
                "source/target pair."
            )
        request_path: str | None = None
        if write_request:
            path = run_dir / "request.yaml"
            atomic_write_text(
                path, yaml.safe_dump(request.model_dump(mode="json"), sort_keys=False)
            )
            request_path = str(path)
        return ResearchNeed(
            level="recommended",
            reasons=reasons,
            scopes=[scope.value for scope in request.scopes],
            topics=[topic.value for topic in request.topics],
            request_path=request_path,
        )

    def start_migration_run(
        self,
        application: Path | str,
        source: str,
        target: str,
        *,
        source_platform: str | None = None,
        target_platform: str | None = None,
        source_endpoint: str | None = None,
        target_endpoint: str | None = None,
        run_id: str | None = None,
        output_dir: Path | str | None = None,
        as_of: date | None = None,
        research: Literal["auto", "skip"] = "auto",
        prompt_sources: Sequence[str] | None = None,
        strict: bool = False,
        defer_prompt_candidates: bool = False,
    ) -> MigrationRunStart:
        """Resolve both models registry-first and prepare one run workspace.

        Nothing is written until both models resolve unambiguously; unresolved
        identifiers return candidates for explicit user confirmation instead.
        Likewise, when the scan detects prompt consumers but resolves no
        prompt source while parseable candidate files exist, the candidates
        are returned for confirmation (retry with `prompt_sources`, or with
        `defer_prompt_candidates=True` to decide on the live run through
        add_prompt_sources) before anything is written.
        """
        as_of = as_of or date.today()
        source_match = self._platform_selected(
            self.match_model(source, source_platform, source_endpoint), "source"
        )
        target_match = self._platform_selected(
            self.match_model(target, target_platform, target_endpoint), "target"
        )
        if (
            source_match.status is not ModelMatchStatus.RESOLVED
            or target_match.status is not ModelMatchStatus.RESOLVED
        ):
            return MigrationRunStart(
                status="needs_confirmation",
                source_match=source_match,
                target_match=target_match,
                next_steps=[
                    "Show the unresolved identifier(s) and candidates to the user and "
                    "ask which model was meant; do not research or guess on their behalf.",
                    "Retry start_migration with the confirmed canonical names (and "
                    "platforms). If a model is genuinely absent from the registry, use "
                    "create_migration_research_request for it instead.",
                ],
            )
        source_resolution = source_match.resolution
        target_resolution = target_match.resolution
        assert source_resolution is not None and target_resolution is not None
        assert source_resolution.platform is not None and target_resolution.platform is not None
        target_derivation = derive_invocation(target_resolution.platform, target)
        if target_derivation.choice is None:
            # The reviewed target platform forbids bare on-demand invocation and
            # several selectors exist: the user must pick one before anything is
            # written. Retrying with the chosen selector's model id seeds it.
            identity = target_resolution.profile.identity
            bare = target_resolution.platform.model_id
            return MigrationRunStart(
                status="needs_confirmation",
                source_match=source_match,
                target_match=ModelMatchResult(
                    query=target,
                    platform_query=target_platform,
                    status=ModelMatchStatus.NEEDS_CONFIRMATION,
                    candidates=[
                        ModelMatchCandidate(
                            canonical_name=identity.canonical_name,
                            display_name=identity.display_name,
                            provider=identity.provider,
                            matched_identifier=selector.model_id,
                            platform=target_resolution.platform.platform,
                            endpoint=target_resolution.platform.endpoint,
                            similarity=1.0,
                            reason=f"invocation selector {selector.name!r}"
                            + (f": {selector.description}" if selector.description else ""),
                        )
                        for selector in target_derivation.selection_required
                    ],
                    notes=[
                        *target_match.notes,
                        f"The bare model id {bare!r} is not invocable on demand on "
                        f"{target_resolution.platform.platform}; an invocation "
                        "selector is required.",
                    ],
                    guidance=(
                        "Ask the user which invocation selector the target should "
                        "use, then retry start_migration with the target set to the "
                        "chosen selector's model id."
                    ),
                ),
                next_steps=[
                    "Show the invocation selector candidates to the user and ask "
                    "which one the deployment should invoke; never choose one on "
                    "their behalf.",
                    "Retry start_migration with the target set to the chosen "
                    "selector's model id (e.g. the full regional inference-profile "
                    "id).",
                ],
            )
        target_choice = target_derivation.choice
        source_derivation = derive_invocation(source_resolution.platform, source)
        source_choice = source_derivation.choice or InvocationChoice(
            invocation_model_id=source_resolution.platform.model_id
        )
        application_path = Path(application).resolve()
        if output_dir is not None:
            run_dir = Path(output_dir).resolve()
            effective_run_id = sanitize_run_id(run_dir.name)
        else:
            effective_run_id = (
                sanitize_run_id(run_id)
                if run_id
                else default_run_id(
                    source_resolution.canonical_name, target_resolution.canonical_name, as_of
                )
            )
            run_dir = default_run_dir(application_path, effective_run_id)
        source_identity = ModelEndpointIdentity(
            provider=source_resolution.identity.provider,
            platform=source_resolution.platform.platform,
            model=source_resolution.canonical_name,
            endpoint=source_resolution.platform.endpoint,
        )
        target_identity = ModelEndpointIdentity(
            provider=target_resolution.identity.provider,
            platform=target_resolution.platform.platform,
            model=target_resolution.canonical_name,
            endpoint=target_resolution.platform.endpoint,
        )
        analysis = self.scan_application(application_path, prompt_sources=prompt_sources)
        candidates = self._unconfirmed_prompt_candidates(analysis)
        if candidates and not defer_prompt_candidates:
            return MigrationRunStart(
                status="needs_confirmation",
                source_match=source_match,
                target_match=target_match,
                prompt_candidates=candidates,
                warnings=list(analysis.warnings),
                next_steps=[
                    f"Prompt discovery found {analysis.prompt_discovery.consumers} prompt "
                    "consumer(s) but resolved no prompt source. Show each candidate in "
                    "prompt_candidates (path, components, evidence) to the user and ask "
                    "which are live prompts for this migration; never choose on their "
                    "behalf. Nothing was written.",
                    "Retry start_migration with prompt_sources set to the confirmed "
                    "paths. If none is a live prompt (or the user wants to decide later), "
                    "retry with defer_prompt_candidates=true: the run then reports "
                    "discovery_incomplete until add_prompt_sources adds the live ones or "
                    "dismisses the rest with the user's rationale.",
                ],
            )
        need = self._research_need(
            analysis,
            run_dir,
            effective_run_id,
            source_identity,
            target_identity,
            as_of,
            write_request=research == "auto",
        )
        config = MigrationRunConfig(
            run_id=effective_run_id,
            application_root=str(application_path),
            source=source_identity,
            target=target_identity,
            source_model_id=source_resolution.platform.model_id,
            target_model_id=target_resolution.platform.model_id,
            source_invocation_model_id=source_choice.invocation_model_id,
            target_invocation_model_id=target_choice.invocation_model_id,
            source_invocation_selector=source_choice.selector,
            target_invocation_selector=target_choice.selector,
            target_invocation_requires_selector=target_choice.requires_selector,
            source_model_spellings=platform_spelling_set(source_resolution.platform),
            target_model_spellings=platform_spelling_set(target_resolution.platform),
            source_model_reference_spellings=model_reference_spellings(
                source_resolution.identity, source_resolution.platform
            ),
            target_model_reference_spellings=model_reference_spellings(
                target_resolution.identity, target_resolution.platform
            ),
            strict=strict,
            created_on=as_of,
            prompt_sources=list(prompt_sources or []),
        )
        write_run_config(run_dir, config)
        next_steps: list[str] = []
        discovery = analysis.prompt_discovery
        if discovery.coverage is not PromptDiscoveryCoverage.RESOLVED:
            next_steps.append(
                f"WARNING: prompt coverage is {discovery.coverage.value} — "
                f"{discovery.dynamic_consumers} prompt consumer(s) have no static source. "
                "get_run_status lists them; confirm each that reads a prompt file with "
                "confirm_prompt_consumer, include unreferenced prompt files with "
                "add_prompt_sources, or dismiss genuinely runtime-built consumers there "
                "with the user's rationale."
            )
        if need.level == "recommended":
            next_steps.extend(
                (
                    "Ask the user whether to run bounded research first (recommended: "
                    + " ".join(need.reasons)
                    + ") or proceed with canonical registry facts as-is.",
                    "If researching: call get_research_prompts(run_dir), run each "
                    "returned researcher/reviewer prompt with a separate agent, then "
                    "call build_session_registry(run_dir).",
                )
            )
        next_steps.extend(
            (
                "Call list_adaptation_tasks(run_dir) ONCE to get the per-file adaptation "
                "worklist; its shared_prompt_guidance applies to every prompt task, and "
                "there is no need to re-list between submissions.",
                "If the worklist reports blockers, call get_blocker_resolutions(run_dir) "
                "and present each blocker's question, options, consequences, and "
                "evidence VERBATIM to the user, one blocker at a time; record each user "
                "answer with record_blocker_decision. Never choose on the user's "
                "behalf, and never retry submissions to make a blocker disappear — "
                "recorded decisions are the only way a blocker is resolved.",
                "For every prompt task, adapt the prompt minimally using its "
                "evidence-linked guidance (each task's verbatim_source is the "
                "UNMODIFIED original, never a proposed adaptation) and call "
                "submit_adapted_prompt with a guidance_dispositions entry for every "
                "guidance item (applied / not_applicable / declined with a note); "
                "for every file task, write the complete adapted file and call "
                "submit_adapted_file with a guidance_dispositions entry for every "
                "required change, or pass unchanged=true when the file needs no "
                "change for the target model — the report then states explicitly "
                "that no change was needed and why.",
                "Document every edit of a changed submission in annotated_changes: "
                "exact original/adapted text anchors, a one-sentence why, and the "
                "evidence behind it (kind 'mechanical' for typo-level fixes); "
                "undocumented or phantom changes are rejected.",
                "Call finalize_migration(run_dir) to write migration-manifest.yaml, "
                "migration-report.md, and the adaptation change log under output/; it "
                "reports any remaining coverage gaps.",
                "Drive the per-change review: get_change_review(run_dir) lists every "
                "annotated change with its why, evidence, before/after spans, and "
                "decision status. Present each pending change VERBATIM and record the "
                "user's accept or reject with record_change_decision — a rejection "
                "deterministically regenerates the deliverable from the remaining "
                "changes, and the user, never you, decides.",
                "Review everything under output/ with the user before applying any "
                "change to the application.",
            )
        )
        return MigrationRunStart(
            status="ready",
            source_match=source_match,
            target_match=target_match,
            run=config,
            paths=run_paths(run_dir),
            research=need,
            warnings=[
                *source_match.notes,
                *target_match.notes,
                *target_choice.warnings,
                *target_choice.notes,
                *source_choice.notes,
                *analysis.warnings,
            ],
            next_steps=next_steps,
        )

    @staticmethod
    def _unconfirmed_prompt_candidates(analysis: ApplicationAnalysis) -> list[PromptCandidate]:
        """Candidates needing start-time confirmation: consumers exist, zero
        resolved sources, and parseable low-confidence candidates are present.

        With unresolved consumers but no candidate there is nothing to
        confirm, so start proceeds (dynamic chat-history apps pay nothing).
        """
        discovery = analysis.prompt_discovery
        if not discovery.consumers or discovery.resolved_sources:
            return []
        return [
            PromptCandidate(
                path=source.path,
                confidence=source.confidence.value,
                components=[item.key or "(whole file)" for item in source.components],
                evidence=list(source.provenance),
            )
            for source in analysis.prompt_sources
            if source.confidence is PromptSourceConfidence.LOW and source.components
        ]

    def get_research_prompts(self, run_dir: Path | str) -> ResearchPromptPack:
        """Scope-isolated researcher and reviewer prompts for a run's request."""
        return render_research_prompts(Path(run_dir))

    def _run_service(
        self,
        run_dir: Path,
        *,
        now: datetime | None = None,
    ) -> tuple[MigrationService, list[str]]:
        """The service for a run: canonical registry plus its session overlay."""
        if (run_dir / "session-manifest.yaml").is_file():
            service, manifest = self.load_session_service(run_dir, as_of=now or datetime.now(UTC))
            return service, manifest_summary_lines(manifest)
        return self, []

    def _plan_for_run(
        self,
        config: MigrationRunConfig,
        run_dir: Path,
        *,
        now: datetime | None = None,
        run_service: tuple[MigrationService, list[str]] | None = None,
        analysis: ApplicationAnalysis | None = None,
    ) -> MigrationPlan:
        """Plan a run over canonical knowledge plus its session overlay when built.

        Recorded blocker decisions are re-applied on every regeneration:
        redesign/accept decisions suppress exactly the live blocker they name
        (injecting the required redesign task), and decisions that no longer
        match a live blocker are reported as stale, never silently applied.

        `run_service` and `analysis` let callers that already resolved the
        session service or scanned the application reuse them instead of
        repeating the work (the scan dominates the latency of the interactive
        blocker-decision loop).
        """
        service, session_lines = run_service or self._run_service(run_dir, now=now)
        if analysis is None:
            analysis = self._scan_for_run(config, service)
        plan = service.generate_migration_plan(
            analysis,
            config.source.model,
            config.target.model,
            source_platform=config.source.platform,
            target_platform=config.target.platform,
            source_endpoint=config.source.endpoint,
            target_endpoint=config.target.endpoint,
        )
        if session_lines:
            plan = plan.model_copy(update={"warnings": [*plan.warnings, *session_lines]})
        plan = apply_prompt_discovery_dismissals(
            plan,
            dismissed_discovery_targets(config),
            [
                f"{item.location.path}:{item.location.line}"
                for item in analysis.findings
                if (item.metadata or {}).get("dismissed")
            ],
        )
        observed = observation_reasons(load_observations(run_dir, config.run_id))
        plan = plan.model_copy(
            update={
                "unknowns": with_run_dir(
                    close_by_action(plan.unknowns, observed), str(Path(run_dir).resolve())
                )
            }
        )
        plan = self._with_invocation_identity(plan, config, service)
        return apply_decisions(plan, load_decision_log(run_dir, config.run_id), config)

    def _with_invocation_identity(
        self,
        plan: MigrationPlan,
        config: MigrationRunConfig,
        service: MigrationService,
    ) -> MigrationPlan:
        """Surface the run's target invocation identity on the plan.

        Canonical identity matching normalizes selector prefixes away, so the
        selector-qualified invocation id is enforced explicitly: the plan
        records the chosen invocation identity (or warns that invocation facts
        are unknown), and a reviewed target platform that forbids bare
        on-demand invocation with no selector recorded gains an
        `invocation_selector_required` blocker — injected before decision
        replay so an accept decision can address it like any other blocker.
        """
        try:
            resolved = service.resolve_model(
                config.target.model, config.target.platform, config.target.endpoint
            )
        except RegistryError:
            return plan
        platform = resolved.platform
        if platform is None:
            return plan
        invocation = platform.invocation
        warnings = list(plan.warnings)
        rationale = list(plan.target_selection_rationale)
        blockers = list(plan.blockers)
        if config.target_invocation_selector and config.target_invocation_model_id:
            rationale.append(
                "Invocation identity: the target is invoked as "
                f"{config.target_invocation_model_id!r} (selector "
                f"{config.target_invocation_selector!r}); deliverables must "
                "reference this id, not the bare platform model id."
            )
        elif invocation is None:
            if config.strict:
                blockers.append(
                    migration_blocker(
                        code="invocation_facts_unknown",
                        category=BlockerCategory.INVOCATION,
                        message=(
                            "STRICT MODE: no reviewed invocation facts exist for the "
                            f"target platform representation {platform.platform}/"
                            f"{platform.model_id}; research the invocation facts (or "
                            "accept the risk explicitly) before deploying."
                        ),
                        data={"side": "target"},
                    )
                )
            else:
                warnings.append(
                    "No reviewed invocation facts exist for "
                    f"{platform.platform}/{platform.model_id}; the bare platform model "
                    "id is used as the invocation id. Verify it is invocable on demand "
                    "before deploying."
                )
        if (
            invocation is not None
            and invocation.bare_on_demand_supported is False
            and config.target_invocation_selector is None
        ):
            names = ", ".join(
                f"{selector.name} ({selector.model_id})" for selector in invocation.selectors
            )
            evidence = sorted(
                {
                    str(source.url)
                    for selector in invocation.selectors
                    for source in selector.sources
                    if source.url
                }
                | {str(source.url) for source in invocation.sources if source.url}
            )
            blockers.append(
                migration_blocker(
                    code="invocation_selector_required",
                    category=BlockerCategory.INVOCATION,
                    message=(
                        "The target platform representation "
                        f"{platform.platform}/{platform.model_id} is not invocable "
                        "by its bare model id; choose an invocation selector: "
                        f"{names}."
                    ),
                    evidence_urls=evidence,
                    data={
                        "selectors": [
                            {"name": selector.name, "model_id": selector.model_id}
                            for selector in invocation.selectors
                        ],
                        "side": "target",
                    },
                )
            )
        if config.strict:
            coverage_issue = next(
                (
                    issue
                    for issue in plan.validation_results
                    if issue.code == "incomplete_prompt_coverage"
                ),
                None,
            )
            if coverage_issue is not None:
                blockers.append(
                    migration_blocker(
                        code="incomplete_prompt_coverage",
                        category=BlockerCategory.OTHER,
                        message="STRICT MODE: " + coverage_issue.message,
                    )
                )
        if (
            warnings == list(plan.warnings)
            and blockers == list(plan.blockers)
            and rationale == list(plan.target_selection_rationale)
        ):
            return plan
        return plan.model_copy(
            update={
                "warnings": warnings,
                "target_selection_rationale": rationale,
                "blockers": blockers,
                "migration_complexity": migration_complexity(
                    blockers,
                    plan.model_differences.highest_severity,
                    plan.required_changes,
                    warnings,
                ),
            }
        )

    def _registry_digest(self) -> str:
        if self.registry_root is None:
            return "in-memory-registry"
        return registry_content_sha256(self.registry_root)

    def _tasks_for_run(
        self,
        config: MigrationRunConfig,
        workspace: Path,
        *,
        now: datetime | None = None,
    ) -> tuple[AdaptationTaskList, set[str], bool]:
        """(worklist with fresh statuses, plan evidence URLs, snapshot reused).

        The persisted snapshot is reused only while its staleness key — a
        content hash over the scanned application, migration.yaml, the blocker
        decision log, the session manifest, and the registry — still matches;
        anything else re-derives the plan and refreshes the snapshot. Task
        statuses are always recomputed from the adaptation log at read time.
        """
        key = snapshot_key(workspace, config, self._registry_digest())
        snapshot = load_snapshot(workspace, config.run_id)
        log = load_adaptation_log(workspace, config.run_id)
        if (
            snapshot is not None
            and snapshot.key == key
            and all((workspace / item).is_file() for item in snapshot.probe_paths)
        ):
            return (
                refresh_task_statuses(snapshot.tasks, log),
                set(snapshot.evidence_urls),
                True,
            )
        run_service = self._run_service(workspace, now=now)
        analysis = self._scan_for_run(config, run_service[0])
        plan = self._plan_for_run(
            config, workspace, now=now, run_service=run_service, analysis=analysis
        )
        base_tasks = derive_adaptation_tasks(
            config, plan, workspace, AdaptationLog(run_id=config.run_id)
        )
        base_tasks = base_tasks.model_copy(
            update={
                "dynamic_prompt_consumers": dynamic_prompt_consumers(
                    analysis, frozenset(out_of_scope_paths(plan))
                )
            }
        )
        registry_evidence = self._run_registry_evidence(config, run_service[0])
        evidence = plan_evidence_urls(plan) | registry_evidence
        save_snapshot(
            workspace,
            WorklistSnapshot(
                run_id=config.run_id,
                key=key,
                tasks=base_tasks,
                evidence_urls=sorted(evidence),
                registry_evidence_urls=sorted(registry_evidence),
                probe_paths=_write_probes(workspace, config, plan),
            ),
        )
        return refresh_task_statuses(base_tasks, log), evidence, False

    @staticmethod
    def _run_registry_evidence(config: MigrationRunConfig, service: MigrationService) -> set[str]:
        """Reviewed source URLs of the run's two models and their pair knowledge.

        Resolved through the run's (session-aware) service, so a selected
        session overlay's accepted-claim sources are included too.
        """
        records: list[Any] = []
        for identity in (config.source, config.target):
            try:
                records.append(
                    service.resolve_model(
                        identity.model, identity.platform, identity.endpoint
                    ).profile
                )
            except RegistryError:
                continue
        records.extend(service.registry.migrations_for(config.source.model, config.target.model))
        return registry_evidence_urls(*records)

    def list_adaptation_tasks(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> AdaptationTaskList:
        """Per-file adaptation worklist derived from the run's migration plan.

        Served from the run's worklist snapshot while its staleness key holds
        (statuses always recomputed from the adaptation log); re-derived and
        re-persisted otherwise.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        tasks, _, _ = self._tasks_for_run(config, workspace, now=now)
        mark_worklist_requested(workspace)
        return tasks

    @staticmethod
    def _predisposed_shared(tasks: AdaptationTaskList, log: AdaptationLog) -> set[str]:
        """Shared guidance ids already disposed by an earlier accepted submission.

        Shared prompt guidance is disposable once per run: the first accepted
        record stands, and later submissions no longer owe those items.
        """
        shared_ids = {item.id for item in tasks.shared_prompt_guidance}
        return {
            disposition.guidance_id
            for entry in log.entries
            if entry.kind == "prompt"
            for disposition in entry.guidance_dispositions
            if disposition.guidance_id in shared_ids
        }

    def _blocker_resolution_context(
        self,
        workspace: Path,
        *,
        now: datetime | None = None,
    ) -> tuple[
        MigrationRunConfig,
        tuple[MigrationService, list[str]],
        ApplicationAnalysis,
        MigrationPlan,
        BlockerResolutionSet,
    ]:
        """One scan, one session-service build, one plan per blocker call."""
        config = load_run_config(workspace)
        run_service = self._run_service(workspace, now=now)
        analysis = self._scan_for_run(config, run_service[0])
        plan = self._plan_for_run(
            config, workspace, now=now, run_service=run_service, analysis=analysis
        )
        resolutions = BlockerResolutionSet(
            run_id=config.run_id,
            resolutions=build_blocker_resolutions(
                run_service[0].registry, config, plan, analysis, str(workspace)
            ),
            decisions=plan.decisions,
            guidance=list(RESOLUTION_GUIDANCE),
        )
        return config, run_service, analysis, plan, resolutions

    def get_blocker_resolutions(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> BlockerResolutionSet:
        """Per-blocker questions with registry-backed options for user decisions."""
        _, _, _, _, resolutions = self._blocker_resolution_context(Path(run_dir), now=now)
        return resolutions

    def record_blocker_decision(
        self,
        run_dir: Path | str,
        blocker_id: str,
        option_id: str,
        rationale: str = "",
        *,
        decided_on: date | None = None,
        now: datetime | None = None,
    ) -> BlockerDecisionResult:
        """Record one user decision for one live blocker; fail closed otherwise.

        Retarget/correction decisions update the run's migration.yaml identity
        immediately (registry-first, so an unknown identity is refused);
        redesign/accept decisions only mark the durable decision, which every
        subsequent plan regeneration re-applies.
        """
        workspace = Path(run_dir)
        config, run_service, analysis, _, resolutions = self._blocker_resolution_context(
            workspace, now=now
        )
        resolution = next(
            (item for item in resolutions.resolutions if item.blocker.id == blocker_id),
            None,
        )
        if resolution is None:
            return BlockerDecisionResult(
                accepted=False,
                problems=[
                    f"blocker {blocker_id!r} is not an unresolved blocker of this run; "
                    "it may already be resolved by a decision, fixed in the "
                    "application, or stale — use the current get_blocker_resolutions "
                    "output"
                ],
                unresolved_blockers=[item.blocker.rendered for item in resolutions.resolutions],
                message="Rejected: unknown or already-resolved blocker id.",
            )
        option = next((item for item in resolution.options if item.id == option_id), None)
        if option is None:
            return BlockerDecisionResult(
                accepted=False,
                problems=[
                    f"option {option_id!r} does not exist for blocker {blocker_id!r}; "
                    "valid options: " + ", ".join(item.id for item in resolution.options)
                ],
                message="Rejected: unknown option id.",
            )
        placeholder_rationale = rationale.strip().startswith("<") and rationale.strip().endswith(
            ">"
        )
        if option.kind is ResolutionKind.ACCEPT_WITH_RATIONALE and (
            not rationale.strip() or placeholder_rationale
        ):
            return BlockerDecisionResult(
                accepted=False,
                problems=[
                    "an accept decision requires the user's own free-text rationale; "
                    "record their words, not a placeholder or template value"
                ],
                message="Rejected: accept-with-rationale requires a rationale.",
            )
        decision = BlockerDecision(
            blocker_id=blocker_id,
            blocker_code=resolution.blocker.code,
            blocker_message=resolution.blocker.message,
            option_id=option.id,
            kind=option.kind,
            summary=option.summary,
            rationale=rationale.strip() or option.summary,
            decided_on=decided_on or date.today(),
            target_change=option.target_change,
            source_change=option.source_change,
            task=option.task,
        )
        service = run_service[0]
        config_updated = False
        if option.kind in {ResolutionKind.RETARGET, ResolutionKind.CORRECTION}:
            # A correction may fix either side; which change is set says which.
            side = "target" if option.target_change is not None else "source"
            change = option.target_change if side == "target" else option.source_change
            assert change is not None
            identity = config.target if side == "target" else config.source
            selector_only = (
                change.invocation_selector is not None
                and change.model is None
                and change.platform is None
                and change.endpoint is None
            )
            try:
                if selector_only:
                    # Only the invocation selector changes; the identity (and its
                    # endpoint) stays exactly what the run already records.
                    resolved = service.resolve_model(
                        identity.model, identity.platform, identity.endpoint
                    )
                elif change.model and change.platform is None:
                    # A model correction must not inherit the mistakenly
                    # declared platform; pin the model's own representation.
                    resolved = service.resolve_model(change.model)
                    if resolved.platform is None:
                        if len(resolved.platforms) != 1:
                            return BlockerDecisionResult(
                                accepted=False,
                                problems=[
                                    f"{change.model!r} has multiple platform "
                                    "representations and none was detected in the "
                                    "application; ask the user which platform the "
                                    f"{side} runs on and restart with start_migration"
                                ],
                                message="Rejected: the corrected identity is ambiguous.",
                            )
                        resolved = service.resolve_model(
                            change.model, resolved.platforms[0].platform
                        )
                else:
                    resolved = service.resolve_model(
                        change.model or identity.model,
                        change.platform or identity.platform,
                        change.endpoint,
                    )
            except RegistryError as exc:
                return BlockerDecisionResult(
                    accepted=False,
                    problems=[f"the chosen identity does not resolve in the registry: {exc}"],
                    message="Rejected: the decision's identity change does not resolve.",
                )
            assert resolved.platform is not None
            new_identity = ModelEndpointIdentity(
                provider=resolved.identity.provider,
                platform=resolved.platform.platform,
                model=resolved.canonical_name,
                endpoint=resolved.platform.endpoint,
            )
            invocation_facts = resolved.platform.invocation
            invocation_id: str | None = None
            selector_chosen: str | None = None
            if change.invocation_selector is not None:
                selector = next(
                    (
                        item
                        for item in (invocation_facts.selectors if invocation_facts else [])
                        if item.name == change.invocation_selector
                    ),
                    None,
                )
                if selector is None:
                    return BlockerDecisionResult(
                        accepted=False,
                        problems=[
                            f"invocation selector {change.invocation_selector!r} is not a "
                            "reviewed selector of the chosen platform representation"
                        ],
                        message="Rejected: the decision names an unknown invocation selector.",
                    )
                invocation_id, selector_chosen = selector.model_id, selector.name
            else:
                derivation = derive_invocation(resolved.platform, change.model)
                if derivation.choice is not None:
                    invocation_id = derivation.choice.invocation_model_id
                    selector_chosen = derivation.choice.selector
                # With several selectors and none derivable, the invocation
                # fields stay unset and the invocation_selector_required
                # blocker surfaces on the next plan regeneration.
            update: dict[str, Any] = {
                side: new_identity,
                f"{side}_model_id": resolved.platform.model_id,
                f"{side}_invocation_model_id": invocation_id,
                f"{side}_invocation_selector": selector_chosen,
                f"{side}_model_spellings": platform_spelling_set(resolved.platform),
                f"{side}_model_reference_spellings": model_reference_spellings(
                    resolved.identity, resolved.platform
                ),
            }
            if side == "target":
                update["target_invocation_requires_selector"] = (
                    invocation_facts is not None
                    and invocation_facts.bare_on_demand_supported is False
                )
            config = config.model_copy(update=update)
            write_run_config(workspace, config)
            config_updated = True
        with run_state_lock(workspace):
            log = upsert_decision(load_decision_log(workspace, config.run_id), decision)
            save_decision_log(workspace, log)
        plan = self._plan_for_run(
            config, workspace, now=now, run_service=run_service, analysis=analysis
        )
        unresolved = [blocker.rendered for blocker in plan.blockers]
        stale = [render_applied_decision(item) for item in plan.decisions if item.status == "stale"]
        next_steps: list[str] = []
        if unresolved and config_updated:
            next_steps.append(
                "The decision changed the run identity, so the plan regenerated; call "
                "get_blocker_resolutions once for the refreshed questions and continue "
                "one blocker at a time."
            )
        elif unresolved:
            next_steps.append(
                "Continue with the remaining blockers from the resolutions already "
                "returned, one at a time; record each user answer with "
                "record_blocker_decision."
            )
        else:
            next_steps.append(
                "Every blocker is resolved. Call list_adaptation_tasks(run_dir) once "
                "(the plan changed), work through the tasks, then "
                "finalize_migration(run_dir)."
            )
        return BlockerDecisionResult(
            accepted=True,
            decision=decision,
            run_config_updated=config_updated,
            unresolved_blockers=unresolved,
            stale_decisions=stale,
            message=(
                f"Decision recorded for blocker {resolution.blocker.code} "
                f"[{blocker_id}]: {option.summary} "
                f"{len(unresolved)} unresolved blocker(s) remain."
            ),
            next_steps=next_steps,
        )

    def _validate_prompt_submission(
        self,
        config: MigrationRunConfig,
        source_path: str,
        adapted_prompt: str,
    ) -> PromptValidationResult:
        """Validate the DECODED runtime prompt values, not the serialized text.

        For structured documents the extracted components are validated as one
        joined payload: serialization tricks cannot hide content from
        validation, aggregate checks (the context-window budget, emptiness)
        run over the full runtime prompt rather than per component, and
        prompt-independent issues are reported once instead of per component.
        Only an unparseable document falls back to the raw text; the workspace
        rejects that syntax error itself.
        """
        format = structured_submission_format(config, source_path)
        text_to_validate = adapted_prompt
        if format is not None and adapted_prompt.strip():
            try:
                components = extract_prompt_components(
                    parse_structured_document(adapted_prompt, format)
                )
            except PromptDocumentError:
                pass
            else:
                text_to_validate = "\n\n".join(component.content for component in components)
        return self.validate_prompt(
            config.target.model,
            text_to_validate,
            source_path=source_path,
            target_platform=config.target.platform,
            target_endpoint=config.target.endpoint,
        )

    def submit_adapted_prompt(
        self,
        run_dir: Path | str,
        source_path: str,
        adapted_prompt: str,
        rationale: str,
        changes: list[str] | None = None,
        *,
        allow_restructure: bool = False,
        unchanged: bool = False,
        guidance_dispositions: Sequence[GuidanceDisposition | Mapping[str, Any]] | None = None,
        annotated_changes: Sequence[AnnotatedChange | Mapping[str, Any]] | None = None,
        default_disposition: Literal["applied", "not_applicable", "declined"] | None = None,
        default_disposition_note: str = "",
        submitted_on: date | None = None,
        now: datetime | None = None,
    ) -> PromptSubmissionResult:
        """Validate and persist one adapted prompt beneath the run's output/prompts/.

        `guidance_dispositions` must dispose every guidance item of the
        prompt's task (its own guidance plus the worklist's shared prompt
        guidance) as applied / not_applicable / declined (with a note);
        submissions that leave guidance undisposed are rejected. Shared
        guidance disposed by an earlier accepted submission is no longer
        owed, and `default_disposition` covers every unlisted item — expanded
        into visibly defaulted per-item records. A changed submission must
        document every edit in `annotated_changes`, each anchored to the
        actual diff with its why and evidence.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        if unchanged and adapted_prompt.strip():
            raise ValueError(
                "unchanged=true cannot be combined with adapted content; omit the "
                "content or drop unchanged"
            )
        dispositions = _coerced_models(
            GuidanceDisposition, guidance_dispositions, "guidance disposition"
        )
        annotations = _coerced_models(AnnotatedChange, annotated_changes, "annotated change")
        if unchanged:
            original = read_original_prompt(config, source_path)
            if original is not None:
                adapted_prompt = original
        validation = self._validate_prompt_submission(config, source_path, adapted_prompt)
        # A blocker the user explicitly accepted must not keep rejecting the
        # prompt at submission: downgrade exactly the accepted issues to
        # warnings so the accept decision is honored end to end.
        validation = downgrade_accepted_prompt_issues(
            validation, load_decision_log(workspace, config.run_id)
        )
        tasks, evidence, _ = self._tasks_for_run(config, workspace, now=now)
        result = workspace_submit_adapted_prompt(
            workspace,
            config,
            source_path,
            adapted_prompt,
            rationale,
            changes or [],
            validation,
            submitted_on or date.today(),
            allow_restructure=allow_restructure,
            unchanged=unchanged,
            guidance_dispositions=dispositions,
            tasks=tasks,
            annotated_changes=annotations,
            known_evidence_urls=evidence,
            default_disposition=default_disposition,
            default_disposition_note=default_disposition_note,
            predisposed=self._predisposed_shared(
                tasks, load_adaptation_log(workspace, config.run_id)
            ),
        )
        return self._with_reapplied_decisions(workspace, config, result)

    def submit_adapted_file(
        self,
        run_dir: Path | str,
        source_path: str,
        adapted_content: str,
        rationale: str,
        changes: list[str] | None = None,
        *,
        new_file: bool = False,
        unchanged: bool = False,
        guidance_dispositions: Sequence[GuidanceDisposition | Mapping[str, Any]] | None = None,
        annotated_changes: Sequence[AnnotatedChange | Mapping[str, Any]] | None = None,
        default_disposition: Literal["applied", "not_applicable", "declined"] | None = None,
        default_disposition_note: str = "",
        submitted_on: date | None = None,
        now: datetime | None = None,
    ) -> FileSubmissionResult:
        """Check and persist one adapted application file beneath output/files/.

        `guidance_dispositions` must dispose every required change of the
        file's task (`default_disposition` covers unlisted items as visibly
        defaulted records), and a changed submission of an existing file must
        document every edit in `annotated_changes`, each anchored to the
        actual diff with its why and evidence.
        """
        if unchanged and adapted_content.strip():
            raise ValueError(
                "unchanged=true cannot be combined with adapted content; omit the "
                "content or drop unchanged"
            )
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        tasks, evidence, _ = self._tasks_for_run(config, workspace, now=now)
        result = workspace_submit_adapted_file(
            workspace,
            config,
            source_path,
            adapted_content,
            rationale,
            changes or [],
            submitted_on or date.today(),
            new_file=new_file,
            unchanged=unchanged,
            guidance_dispositions=_coerced_models(
                GuidanceDisposition, guidance_dispositions, "guidance disposition"
            ),
            tasks=tasks,
            annotated_changes=_coerced_models(
                AnnotatedChange, annotated_changes, "annotated change"
            ),
            known_evidence_urls=evidence,
            default_disposition=default_disposition,
            default_disposition_note=default_disposition_note,
        )
        return self._with_reapplied_decisions(workspace, config, result)

    def submit_adaptations(
        self,
        run_dir: Path | str,
        submissions: Sequence[AdaptationSubmission | Mapping[str, Any]],
        *,
        submitted_on: date | None = None,
        now: datetime | None = None,
    ) -> BatchSubmissionResult:
        """Validate every submission independently; apply the accepted subset once.

        Per-item accept/reject results, never all-or-nothing: rejected items
        change nothing, and every accepted deliverable lands in ONE locked,
        atomic write to changes.yaml. Duplicate source paths within a batch
        keep the last accepted item. Shared prompt guidance disposed by an
        earlier item (or an earlier run submission) is no longer owed by
        later items.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        items = _coerced_models(AdaptationSubmission, submissions, "adaptation submission")
        tasks, evidence, _ = self._tasks_for_run(config, workspace, now=now)
        decision_log = load_decision_log(workspace, config.run_id)
        predisposed = self._predisposed_shared(tasks, load_adaptation_log(workspace, config.run_id))
        shared_ids = {item.id for item in tasks.shared_prompt_guidance}
        when = submitted_on or date.today()
        results: list[PromptSubmissionResult | FileSubmissionResult] = []
        pending: list[PendingSubmission] = []
        for item in items:
            if item.unchanged and item.content.strip():
                rejection = (
                    "unchanged=true cannot be combined with adapted content; omit the "
                    "content or drop unchanged"
                )
                if item.kind == "prompt":
                    results.append(
                        PromptSubmissionResult(
                            accepted=False,
                            source_path=item.source_path,
                            validation=PromptValidationResult(
                                valid=False,
                                source_prompt_sha256=hashlib.sha256(
                                    item.content.encode("utf-8")
                                ).hexdigest(),
                                target_model=config.target.model,
                                issues=[],
                            ),
                            message=f"Rejected: {rejection}",
                        )
                    )
                else:
                    results.append(
                        FileSubmissionResult(
                            accepted=False,
                            source_path=item.source_path,
                            problems=[rejection],
                            message=f"Rejected: {rejection}",
                        )
                    )
                continue
            if item.kind == "prompt":
                content = item.content
                if item.unchanged:
                    original = read_original_prompt(config, item.source_path)
                    if original is not None:
                        content = original
                validation = downgrade_accepted_prompt_issues(
                    self._validate_prompt_submission(config, item.source_path, content),
                    decision_log,
                )
                prompt_result, prompt_pending = prepare_adapted_prompt(
                    workspace,
                    config,
                    item.source_path,
                    content,
                    item.rationale,
                    list(item.changes),
                    validation,
                    when,
                    allow_restructure=item.allow_restructure,
                    unchanged=item.unchanged,
                    guidance_dispositions=list(item.guidance_dispositions),
                    tasks=tasks,
                    annotated_changes=list(item.annotated_changes),
                    known_evidence_urls=evidence,
                    default_disposition=item.default_disposition,
                    default_disposition_note=item.default_disposition_note,
                    predisposed=predisposed,
                )
                results.append(prompt_result)
                if prompt_pending is not None:
                    pending.append(prompt_pending)
                    predisposed |= {
                        disposition.guidance_id
                        for disposition in prompt_pending.entry.guidance_dispositions
                        if disposition.guidance_id in shared_ids
                    }
            else:
                file_result, file_pending = prepare_adapted_file(
                    workspace,
                    config,
                    item.source_path,
                    item.content,
                    item.rationale,
                    list(item.changes),
                    when,
                    new_file=item.new_file,
                    unchanged=item.unchanged,
                    guidance_dispositions=list(item.guidance_dispositions),
                    tasks=tasks,
                    annotated_changes=list(item.annotated_changes),
                    known_evidence_urls=evidence,
                    default_disposition=item.default_disposition,
                    default_disposition_note=item.default_disposition_note,
                )
                results.append(file_result)
                if file_pending is not None:
                    pending.append(file_pending)
        apply_submissions(workspace, config, pending)
        reapplied: list[PromptSubmissionResult | FileSubmissionResult] = []
        for outcome in results:
            if not outcome.accepted:
                reapplied.append(outcome)
            elif isinstance(outcome, PromptSubmissionResult):
                reapplied.append(self._with_reapplied_decisions(workspace, config, outcome))
            else:
                reapplied.append(self._with_reapplied_decisions(workspace, config, outcome))
        results = reapplied
        accepted = sum(item.accepted for item in results)
        return BatchSubmissionResult(
            run_id=config.run_id,
            results=results,
            accepted=accepted,
            message=(
                f"{accepted} of {len(results)} submission(s) accepted and applied in "
                "one atomic write; rejected items changed nothing (see per-item "
                "results)."
            ),
        )

    def record_change_decisions(
        self,
        run_dir: Path | str,
        decisions: Sequence[ChangeDecisionRequest | Mapping[str, Any]],
        *,
        decided_on: date | None = None,
    ) -> BatchChangeDecisionResult:
        """Record several review decisions in order, with per-decision results.

        Decisions apply sequentially (each regeneration reflects every earlier
        rejection), never all-or-nothing: a refused decision is reported and
        the rest continue.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        items = _coerced_models(ChangeDecisionRequest, decisions, "change decision")
        results = [
            review_record_change_decision(
                workspace,
                config,
                item.source_path,
                item.change_id,
                item.decision,
                note=item.note,
                decided_on=decided_on,
            )
            for item in items
        ]
        recorded = sum(item.accepted for item in results)
        return BatchChangeDecisionResult(
            run_id=config.run_id,
            results=results,
            recorded=recorded,
            message=(
                f"{recorded} of {len(results)} decision(s) recorded; refused decisions "
                "changed nothing (see per-decision results)."
            ),
        )

    def get_run_status(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> RunStatus:
        """The run's state machine position with the single next action.

        Derived from the worklist snapshot (re-derived only when stale), the
        adaptation log, and the decision logs, so status checks stay cheap.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        # The run moved past the research question only on explicit host
        # activity: the worklist was requested (list_adaptation_tasks writes
        # the marker), a blocker decision was recorded, or a submission
        # landed (the log check below). The snapshot file proves nothing —
        # this very call persists it as a cache, and inferring from it made
        # a second status call silently steer past research (v1.5.0 defect).
        moved_past_research = (
            worklist_requested(workspace) or (workspace / DECISIONS_FILENAME).is_file()
        )
        tasks, _, reused = self._tasks_for_run(config, workspace, now=now)
        log = load_adaptation_log(workspace, config.run_id)
        change_decisions = load_change_decision_log(workspace, config.run_id)
        gaps = coverage_gaps(tasks)
        pending_review = 0
        for entry in log.entries:
            if entry_is_unchanged(entry) or not entry.annotated_changes:
                continue
            decided = {
                item.change_id
                for item in change_decisions.decisions
                if decision_matches_entry(item, entry)
            }
            pending_review += sum(change.id not in decided for change in entry.annotated_changes)
        research_pending: list[str] = []
        if (
            (workspace / "request.yaml").is_file()
            and not (workspace / "session-manifest.yaml").is_file()
            and not moved_past_research
            and not log.entries
        ):
            pack = self.get_research_prompts(workspace)
            research_pending = [
                scope.scope.value for scope in pack.scopes if scope.status != "complete"
            ]
        if research_pending:
            state: Literal[
                "research_pending",
                "discovery_incomplete",
                "blockers_pending",
                "tasks_pending",
                "review_pending",
                "ready_to_finalize",
            ] = "research_pending"
            next_action = (
                "Research is recommended but optional: ask the user, then either run "
                "the get_research_prompts stages (scopes pending: "
                + ", ".join(research_pending)
                + ") and build_session_registry, or proceed to list_adaptation_tasks."
            )
        elif tasks.prompt_coverage != "resolved" and (tasks.prompt_candidates or config.strict):
            state = "discovery_incomplete"
            next_action = _discovery_action(str(run_dir), tasks, strict=config.strict)
        elif tasks.blockers:
            state = "blockers_pending"
            next_action = (
                "Call get_blocker_resolutions(run_dir) and present each blocker's "
                "question, options, and evidence VERBATIM, one at a time; record each "
                "user answer with record_blocker_decision."
            )
        elif gaps or tasks.unaffected_files:
            state = "tasks_pending"
            next_action = (
                f"Work the worklist: {len(gaps)} task(s) still need a submission"
                + (
                    f" and {len(tasks.unaffected_files)} unaffected file(s) need one "
                    "confirm_unaffected call"
                    if tasks.unaffected_files
                    else ""
                )
                + "."
            )
        elif pending_review:
            state = "review_pending"
            next_action = (
                f"{pending_review} annotated change(s) await user decisions: present "
                "each pending change from get_change_review VERBATIM and record the "
                "user's accept/reject with record_change_decision(s)."
            )
        else:
            state = "ready_to_finalize"
            validation = load_validation_disposition(workspace, config.run_id)
            next_action = (
                "Validation is a two-pass flow: call finalize_migration(run_dir) to write "
                "the deliverables and the generated contract test; the user runs that "
                "test (or a BYOK evaluation via generate_eval_suite / run_migration_eval); "
                "record the outcome with record_validation_disposition; then call "
                "finalize_migration again."
                if validation is None
                else "Validation is recorded; call finalize_migration(run_dir) to "
                "write the final deliverables."
            )
        if state in {"tasks_pending", "review_pending", "ready_to_finalize"} and tasks.unknowns:
            next_action += (
                f" {len(tasks.unknowns)} open unknown(s) each carry their exact next action "
                "(list_adaptation_tasks `unknowns`); ask the user before acting on one, and "
                "record empirical results with record_observation."
            )
        if state in {"tasks_pending", "review_pending", "ready_to_finalize"} and (
            tasks.prompt_coverage != "resolved"
        ):
            next_action = (
                f"WARNING: prompt coverage is {tasks.prompt_coverage} — prompt adaptation "
                "is incomplete. "
                + (
                    "Unreferenced candidate prompt file(s): "
                    + ", ".join(tasks.prompt_candidates)
                    + "; ask the user which are live prompts and include them with "
                    "add_prompt_sources. "
                    if tasks.prompt_candidates
                    else f"{len(tasks.dynamic_prompt_consumers)} prompt consumer(s) have no "
                    "static source; confirm each with confirm_prompt_consumer or dismiss it "
                    "with the user's rationale (add_prompt_sources dismiss=...). "
                )
                + next_action
            )
        return RunStatus(
            run_id=config.run_id,
            state=state,
            next_action=next_action,
            unresolved_blockers=len(tasks.blockers),
            pending_tasks=len(gaps),
            unconfirmed_unaffected=len(tasks.unaffected_files),
            pending_review_changes=pending_review,
            research_scopes_pending=research_pending,
            prompt_coverage=tasks.prompt_coverage,
            prompt_candidates=tasks.prompt_candidates,
            dynamic_prompt_consumers=len(tasks.dynamic_prompt_consumers),
            open_unknowns=len(tasks.unknowns),
            snapshot_reused=reused,
        )

    def record_observation(
        self,
        run_dir: Path | str,
        subject: str,
        outcome: str,
        evidence: str,
        *,
        now: datetime | None = None,
    ) -> ObservationResult:
        """Record the target's observed behavior for one plan unknown.

        `subject` is the unknown's id (from the report, the worklist
        `unknowns`, or a probe script's docstring). The observation is
        RUN-SCOPED: it closes that unknown in this run's plan and renders in
        the report; it never touches the registry. A later observation for
        the same unknown replaces the earlier one.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        plan = self._plan_for_run(config, workspace, now=now)
        by_id = {item.id: item for item in plan.unknowns}
        problems: list[str] = []
        unknown = by_id.get(subject.strip())
        if unknown is None:
            problems.append(
                f"{subject!r} is not the id of an unknown in this run's plan (ids look "
                "like 'contested_evidence:0123456789')"
            )
        elif unknown.status == "closed_by_scan":
            problems.append(
                f"{subject!r} was already answered by the scan: {unknown.closed_reason}"
            )
        if not outcome.strip():
            problems.append("an observation needs the observed outcome")
        if not evidence.strip():
            problems.append("an observation needs its evidence (printed result, request id, URL)")
        if problems or unknown is None:
            return ObservationResult(
                run_id=config.run_id,
                accepted=False,
                problems=problems,
                open_unknowns=sum(item.is_open for item in plan.unknowns),
                message="Rejected: nothing was recorded. " + "; ".join(problems),
            )
        observation = RunObservation(
            unknown_id=unknown.id,
            subject=unknown.subject,
            outcome=outcome.strip(),
            evidence=evidence.strip(),
            recorded_on=(now or datetime.now(UTC)).date(),
        )
        with run_state_lock(workspace):
            log = upsert_observation(load_observations(workspace, config.run_id), observation)
            save_observations(workspace, log)
        remaining = sum(item.is_open and item.id != unknown.id for item in plan.unknowns)
        return ObservationResult(
            run_id=config.run_id,
            accepted=True,
            observation=observation,
            open_unknowns=remaining,
            message=(
                f"Observation recorded for {unknown.subject} [{unknown.id}] (run-scoped; the "
                "registry is unchanged — promoting it is a separate propose_registry_update). "
                f"{remaining} unknown(s) remain open."
            ),
        )

    def add_prompt_sources(
        self,
        run_dir: Path | str,
        paths: Sequence[str],
        *,
        dismiss: Sequence[str] = (),
        rationale: str = "",
        now: datetime | None = None,
    ) -> PromptDiscoveryUpdate:
        """Add prompt sources to a live run, or dismiss candidates/consumers.

        Updates `migration.yaml` under the run lock, which changes the
        worklist staleness key: the worklist re-derives, already-submitted
        deliverables keep their entries, and newly covered prompt files appear
        as pending prompt tasks. A dismissal names current unreferenced
        candidate files or dynamic consumer `path:line` addresses and needs
        the user's rationale; it is recorded, never silent. All-or-nothing:
        any problem writes nothing.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        before, _, _ = self._tasks_for_run(config, workspace, now=now)
        resolver = application_path_resolver(config.application_root)
        problems: list[str] = []
        if not paths and not dismiss:
            problems.append("nothing to add or dismiss: pass paths and/or dismiss")
        if dismiss and not rationale.strip():
            problems.append("a dismissal needs the user's rationale (why these are not prompts)")
        added: list[str] = []
        for raw in paths:
            resolved = resolve_override(raw, resolver)
            if resolved is None:
                problems.append(f"{raw!r} is not a file inside the application")
            elif source_format(resolved) is None:
                problems.append(f"{raw!r} is not a supported prompt file format")
            elif resolved not in config.prompt_sources and resolved not in added:
                added.append(resolved)
        dismissable = {
            *before.prompt_candidates,
            *(item.location for item in before.dynamic_prompt_consumers),
        }
        for target in dismiss:
            if target not in dismissable:
                problems.append(
                    f"{target!r} is neither a current unreferenced candidate prompt file nor "
                    "a dynamic prompt consumer address (path:line) of this run"
                )
        if problems:
            return self._discovery_update(
                config,
                before,
                accepted=False,
                problems=problems,
                message="Rejected: nothing was recorded. " + "; ".join(problems),
            )
        decided_on = (now or datetime.now(UTC)).date()
        with run_state_lock(workspace):
            config = load_run_config(workspace)
            update: dict[str, Any] = {"prompt_sources": [*config.prompt_sources, *added]}
            if dismiss:
                update["prompt_discovery_dismissals"] = [
                    *config.prompt_discovery_dismissals,
                    PromptDiscoveryDismissal(
                        targets=sorted(set(dismiss)),
                        rationale=rationale.strip(),
                        decided_on=decided_on,
                    ),
                ]
            config = config.model_copy(update=update)
            write_run_config(workspace, config)
        after, _, _ = self._tasks_for_run(config, workspace, now=now)
        return self._discovery_update(
            config,
            after,
            accepted=True,
            added=added,
            dismissed=sorted(set(dismiss)),
            before=before,
            message=(
                f"Recorded: {len(added)} prompt source(s) added, {len(set(dismiss))} "
                "target(s) dismissed. The worklist re-derived; submitted deliverables "
                "keep their entries."
            ),
        )

    def confirm_prompt_consumer(
        self,
        run_dir: Path | str,
        location: str,
        source_path: str,
        *,
        now: datetime | None = None,
    ) -> PromptDiscoveryUpdate:
        """Record that one dynamic prompt consumer reads one prompt file.

        The consumer (`path:line`, as listed by get_run_status /
        list_adaptation_tasks) becomes source-backed and the file becomes a
        prompt source with the confirmation as its provenance. Recorded in
        `migration.yaml` under the run lock; the worklist re-derives.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        before, _, _ = self._tasks_for_run(config, workspace, now=now)
        resolver = application_path_resolver(config.application_root)
        problems: list[str] = []
        known = {item.location for item in before.dynamic_prompt_consumers} | set(
            consumer_confirmation_map(config)
        )
        if location not in known:
            problems.append(
                f"{location!r} is not a dynamic prompt consumer of this run (use a "
                "path:line address from dynamic_prompt_consumers)"
            )
        resolved = resolve_override(source_path, resolver)
        if resolved is None:
            problems.append(f"{source_path!r} is not a file inside the application")
        elif source_format(resolved) is None:
            problems.append(f"{source_path!r} is not a supported prompt file format")
        if problems or resolved is None:
            return self._discovery_update(
                config,
                before,
                accepted=False,
                problems=problems,
                message="Rejected: nothing was recorded. " + "; ".join(problems),
            )
        with run_state_lock(workspace):
            config = load_run_config(workspace)
            confirmations = [
                item for item in config.prompt_consumer_confirmations if item.location != location
            ]
            confirmations.append(
                PromptConsumerConfirmation(
                    location=location,
                    source_path=resolved,
                    decided_on=(now or datetime.now(UTC)).date(),
                )
            )
            config = config.model_copy(update={"prompt_consumer_confirmations": confirmations})
            write_run_config(workspace, config)
        after, _, _ = self._tasks_for_run(config, workspace, now=now)
        return self._discovery_update(
            config,
            after,
            accepted=True,
            confirmed=[location],
            added=[resolved]
            if resolved not in {t.source_path for t in before.prompt_tasks}
            else [],
            before=before,
            message=(
                f"Recorded: prompt consumer {location} reads {resolved}. The worklist "
                "re-derived; submitted deliverables keep their entries."
            ),
        )

    @staticmethod
    def _discovery_update(
        config: MigrationRunConfig,
        tasks: AdaptationTaskList,
        *,
        accepted: bool,
        message: str,
        added: Sequence[str] = (),
        confirmed: Sequence[str] = (),
        dismissed: Sequence[str] = (),
        problems: Sequence[str] = (),
        before: AdaptationTaskList | None = None,
    ) -> PromptDiscoveryUpdate:
        previous = {item.source_path for item in before.prompt_tasks} if before else set()
        return PromptDiscoveryUpdate(
            run_id=config.run_id,
            accepted=accepted,
            added_sources=list(added),
            confirmed_consumers=list(confirmed),
            dismissed=list(dismissed),
            new_prompt_tasks=sorted(
                item.source_path for item in tasks.prompt_tasks if item.source_path not in previous
            )
            if before
            else [],
            prompt_coverage=tasks.prompt_coverage,
            prompt_candidates=tasks.prompt_candidates,
            dynamic_prompt_consumers=[item.location for item in tasks.dynamic_prompt_consumers],
            problems=list(problems),
            message=message,
        )

    def _with_reapplied_decisions(
        self,
        workspace: Path,
        config: MigrationRunConfig,
        result: _SubmissionResultT,
    ) -> _SubmissionResultT:
        """Keep the deliverable consistent with live rejections after a submit.

        A resubmission with identical content fingerprints keeps its recorded
        change decisions live; the freshly written deliverable must re-apply
        those rejections instead of silently reverting to the full submission.
        """
        if not result.accepted:
            return result
        applied, problems = reapply_change_decisions(workspace, config, result.source_path)
        if applied:
            return result.model_copy(
                update={
                    "message": result.message
                    + f" {applied} previously recorded rejection(s) for this file "
                    "still apply and were re-applied to the deliverable."
                }
            )
        if problems:
            return result.model_copy(
                update={
                    "message": result.message
                    + " WARNING: previously recorded rejections could not be "
                    "re-applied to the deliverable: " + "; ".join(problems)
                }
            )
        return result

    def get_change_review(self, run_dir: Path | str) -> ChangeReviewSet:
        """Per-deliverable annotated changes paired with their decision state.

        Presents each change (why, evidence, before/after spans) for the user
        to accept or reject individually; the decoded unified diff of every
        reviewable deliverable comes along for context. Reviews whose
        application file drifted since submission are marked stale.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        _, evidence, _ = self._tasks_for_run(config, workspace)
        snapshot = load_snapshot(workspace, config.run_id)
        registry_evidence = set(snapshot.registry_evidence_urls) if snapshot else set()
        return build_change_review(
            workspace,
            config,
            known_evidence_urls=evidence,
            registry_evidence_urls=registry_evidence,
        )

    def record_change_decision(
        self,
        run_dir: Path | str,
        source_path: str,
        change_id: str,
        decision: str,
        note: str = "",
        *,
        decided_on: date | None = None,
    ) -> ChangeDecisionResult:
        """Record one accept/reject decision and regenerate the deliverable.

        Decisions are durable in change-decisions.yaml and keyed to the
        submission fingerprints; the deliverable under output/ is rebuilt
        deterministically from the original content, the as-submitted
        content, and every live rejection — or the decision is refused.
        """
        workspace = Path(run_dir)
        return review_record_change_decision(
            workspace,
            load_run_config(workspace),
            source_path,
            change_id,
            decision,
            note=note,
            decided_on=decided_on,
        )

    def confirm_unaffected(
        self,
        run_dir: Path | str,
        paths: Sequence[str],
        rationale: str,
        *,
        acknowledge_source_references: bool = False,
        submitted_on: date | None = None,
        now: datetime | None = None,
    ) -> UnaffectedConfirmation:
        """Close every listed unaffected file with one reviewed no-change entry.

        Each path is validated independently (per-file accept/reject): it must
        be on the worklist's `unaffected_files` list — a file with required
        changes needs a real submission, and a file already confirmed or
        outside the worklist is refused. Every accepted path passes the
        existing unchanged guards and lands in `changes.yaml` exactly like an
        individual unchanged submission.

        `acknowledge_source_references=True` additionally accepts application
        files the finalize sweep flagged (`source_reference_uncovered`: they
        name the source model but are not worklist tasks) whose reference is
        intentional — documentation or history. The rationale must be the
        user's own; entries are marked acknowledged and reported as such.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        if not rationale.strip():
            raise ValueError(
                "confirm_unaffected requires a rationale recording the review that "
                "found these files unaffected"
            )
        tasks, _, _ = self._tasks_for_run(config, workspace, now=now)
        allowed = set(tasks.unaffected_files)
        task_paths = {task.source_path for task in tasks.file_tasks} | {
            task.source_path for task in tasks.prompt_tasks
        }
        logged = {
            entry.source_path for entry in load_adaptation_log(workspace, config.run_id).entries
        }
        results: list[FileSubmissionResult] = []
        for path in dict.fromkeys(paths):
            if (
                acknowledge_source_references
                and path not in allowed
                and path not in task_paths
                and path not in logged
                and self._names_source_model(config, path)
            ):
                results.append(
                    workspace_submit_adapted_file(
                        workspace,
                        config,
                        path,
                        "",
                        rationale,
                        ["Intentional source-model reference acknowledged by the user."],
                        submitted_on or date.today(),
                        unchanged=True,
                        tasks=tasks,
                        acknowledge_source_reference=True,
                    )
                )
            elif path in allowed:
                results.append(
                    workspace_submit_adapted_file(
                        workspace,
                        config,
                        path,
                        "",
                        rationale,
                        [],
                        submitted_on or date.today(),
                        unchanged=True,
                        tasks=tasks,
                    )
                )
            elif path in task_paths:
                results.append(
                    FileSubmissionResult(
                        accepted=False,
                        source_path=path,
                        problems=[
                            "this file has its own adaptation task with required "
                            "changes; submit it through submit_adapted_file or "
                            "submit_adapted_prompt instead"
                        ],
                        message="Rejected: the file has required changes.",
                    )
                )
            else:
                results.append(
                    FileSubmissionResult(
                        accepted=False,
                        source_path=path,
                        problems=[
                            "this path is not on the worklist's unaffected_files list "
                            "(already confirmed, or not part of this run)"
                        ],
                        message="Rejected: not an unconfirmed unaffected file.",
                    )
                )
        confirmed = sum(item.accepted for item in results)
        rejected = len(results) - confirmed
        return UnaffectedConfirmation(
            run_id=config.run_id,
            results=results,
            confirmed=confirmed,
            message=(
                f"{confirmed} unaffected file(s) recorded as reviewed no-change "
                f"entries; {rejected} path(s) rejected (see per-file results)."
            ),
        )

    @staticmethod
    def _names_source_model(config: MigrationRunConfig, path: str) -> bool:
        """Whether an application file inside the run's scan set names the source."""
        root = Path(config.application_root)
        if not root.is_dir():
            return False
        python_files, candidate_files = scannable_files(root)
        scanned = {
            item.relative_to(root).as_posix(): item for item in (*python_files, *candidate_files)
        }
        target = scanned.get(path)
        if target is None:
            return False
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False
        return any(spelling in text for spelling in source_detection_spellings(config))

    def record_validation_disposition(
        self,
        run_dir: Path | str,
        method: Literal["byok_evaluation", "generated_tests", "accepted_without_validation"],
        rationale: str = "",
        *,
        outcome: Literal["run_passed", "run_failed", "not_run"] | None = None,
        outcome_summary: str = "",
        evaluation_run_path: str | None = None,
        decided_on: date | None = None,
    ) -> ValidationDisposition:
        """Record how this run's migration was validated; durable per run.

        `accepted_without_validation` requires the user's own free-text
        rationale — it is never a default and never chosen by an agent.
        `generated_tests` requires the attested outcome `run_passed` plus the
        test runner's summary line; `byok_evaluation` requires the evaluation
        run artifact, whose suite must be bound to this run's finalized
        manifest. The toolkit never runs either itself.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        placeholder = rationale.strip().startswith("<") and rationale.strip().endswith(">")
        if method == "accepted_without_validation" and (not rationale.strip() or placeholder):
            raise ValueError(
                "accepting a migration without validation requires the user's own "
                "free-text rationale; record their words, not a placeholder"
            )
        bound_hash: str | None = None
        if method == "generated_tests":
            if outcome != "run_passed" or not outcome_summary.strip():
                raise ValueError(
                    "generated_tests requires outcome='run_passed' and the test runner's "
                    "summary line (for example '1 passed in 0.12s') from a run the user "
                    "executed after wiring build_request(); a failing or unrun test is "
                    "not validation — fix and re-run it, validate with a BYOK "
                    "evaluation, or record accepted_without_validation with the user's "
                    "own rationale"
                )
        elif method == "byok_evaluation":
            bound_hash = self._evaluation_binding(workspace, evaluation_run_path)
        disposition = ValidationDisposition(
            run_id=config.run_id,
            method=method,
            rationale=rationale.strip(),
            decided_on=decided_on or date.today(),
            outcome=outcome if method == "generated_tests" else None,
            outcome_summary=outcome_summary.strip() if method == "generated_tests" else "",
            evaluation_run_path=evaluation_run_path if method == "byok_evaluation" else None,
            bound_manifest_sha256=bound_hash,
        )
        save_validation_disposition(workspace, disposition)
        return disposition

    @staticmethod
    def _finalized_manifest_hash(workspace: Path) -> str | None:
        path = Path(run_paths(workspace).manifest_path)
        if not path.is_file():
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "migration" in raw:
            raw = raw["migration"]
        return manifest_sha256(MigrationPlan.model_validate(raw))

    def _evaluation_binding(self, workspace: Path, evaluation_run_path: str | None) -> str:
        """The manifest hash a BYOK evaluation run is bound to, verified."""
        if not evaluation_run_path:
            raise ValueError(
                "byok_evaluation requires evaluation_run_path: the MigrationEvalRun YAML "
                "written by run_migration_eval for this run's finalized manifest"
            )
        candidate = Path(evaluation_run_path)
        path = candidate if candidate.is_absolute() else workspace / candidate
        if not path.is_file():
            raise ValueError(f"evaluation run artifact not found: {path}")
        run = MigrationEvalRun.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        current = self._finalized_manifest_hash(workspace)
        if current is None:
            raise ValueError(
                "no finalized manifest exists yet: call finalize_migration first; the "
                "evaluation suite binds to output/migration-manifest.yaml"
            )
        if run.suite.migration_manifest_sha256 != current:
            raise ValueError(
                "the evaluation run is bound to a different manifest than this run's "
                "current output/migration-manifest.yaml (a later decision or submission "
                "changed the plan); regenerate the suite from the current manifest and "
                "re-run the evaluation"
            )
        return current

    def finalize_migration_run(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> MigrationRunFinalization:
        """Write the manifest, the report with per-file changes/rationale, and gaps.

        Also emits the deterministic contract-test deliverable under
        output/validation/ (supported invocation mappings only) and, in strict
        mode, lists every unmet strict requirement as a violation.
        """
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        run_service = self._run_service(workspace, now=now)
        analysis = self._scan_for_run(config, run_service[0])
        plan = self._plan_for_run(
            config, workspace, now=now, run_service=run_service, analysis=analysis
        )
        log = load_adaptation_log(workspace, config.run_id)
        paths = run_paths(workspace)
        Path(paths.output_dir).mkdir(parents=True, exist_ok=True)
        atomic_write_text(paths.manifest_path, self.migration_manifest_as_yaml(plan))
        tasks = derive_adaptation_tasks(config, plan, workspace, log)
        gaps = coverage_gaps(tasks)
        change_decisions = load_change_decision_log(workspace, config.run_id)
        consistency = check_cross_surface_consistency(
            workspace,
            config,
            analysis,
            log,
            accounted_paths={
                *(task.source_path for task in tasks.prompt_tasks),
                *(task.source_path for task in tasks.file_tasks),
                *tasks.unaffected_files,
                *out_of_scope_paths(plan),
            },
        )
        validation_disposition = load_validation_disposition(workspace, config.run_id)
        validation_problem = _validation_problem(validation_disposition, manifest_sha256(plan))
        contract_test_path: str | None = None
        if plan.invocation_changes:
            contract_test = render_contract_test(config, plan.invocation_changes[0])
            if contract_test is not None:
                contract_target = workspace / CONTRACT_TEST_RELATIVE_PATH
                atomic_write_text(contract_target, contract_test)
                contract_test_path = str(contract_target)
        probe_paths = _write_probes(workspace, config, plan)
        observations = load_observations(workspace, config.run_id)
        undecided_changes: list[str] = []
        for entry in log.entries:
            if entry_is_unchanged(entry) or not entry.annotated_changes:
                continue
            decided = {
                item.change_id
                for item in change_decisions.decisions
                if decision_matches_entry(item, entry)
            }
            pending = [change.id for change in entry.annotated_changes if change.id not in decided]
            if pending:
                undecided_changes.append(
                    f"{entry.source_path}: {len(pending)} change(s) pending review"
                )
        report = _with_action_required(
            self.migration_report(plan),
            _action_required_lines(
                plan,
                gaps=gaps,
                unaffected=list(tasks.unaffected_files),
                undecided=undecided_changes,
                consistency=[finding.rendered for finding in consistency],
                validation_disposition=validation_disposition,
                validation_problem=validation_problem,
            ),
        )
        report = (
            report.rstrip("\n")
            + "\n\n"
            + render_adaptation_section(log, gaps, change_decisions)
            + "\n"
            + render_consistency_section(consistency)
            + "\n"
            + _render_validation_section(
                validation_disposition, contract_test_path, validation_problem
            )
            + "\n"
            + render_observations_section(observations)
        )
        atomic_write_text(paths.report_path, report)
        adapted_prompts = sum(
            entry.kind == "prompt" and not entry_is_unchanged(entry) for entry in log.entries
        )
        adapted_files = sum(
            entry.kind == "file" and not entry_is_unchanged(entry) for entry in log.entries
        )
        reviewed_unchanged = sum(entry_is_unchanged(entry) for entry in log.entries)
        # Superseded retarget/correction decisions were honored and later
        # replaced; they belong with the resolved history, never stale alarms.
        resolved = [
            render_applied_decision(item)
            for item in plan.decisions
            if item.status in {"applied", "superseded"}
        ]
        stale = [render_applied_decision(item) for item in plan.decisions if item.status == "stale"]
        blocker_note = (
            f"{len(plan.blockers)} blocker(s) remain unresolved; drive them to "
            "user decisions with get_blocker_resolutions/record_blocker_decision."
            if plan.blockers
            else (
                f"every blocker is resolved ({len(resolved)} by recorded decision)."
                if resolved
                else "no blockers."
            )
        )
        if stale:
            blocker_note += f" {len(stale)} recorded decision(s) are STALE and were not applied."
        strict_violations: list[str] = []
        if config.strict:
            strict_violations.extend(
                f"coverage gap: {gap} has no adaptation deliverable" for gap in gaps
            )
            strict_violations.extend(
                f"unconfirmed unaffected file: {path}" for path in tasks.unaffected_files
            )
            strict_violations.extend(
                f"consistency finding: {finding.rendered}" for finding in consistency
            )
            strict_violations.extend(
                f"undecided annotated changes: {item}" for item in undecided_changes
            )
            if validation_disposition is None:
                strict_violations.append(
                    "no validation disposition is recorded (record_validation_disposition: "
                    "byok_evaluation, generated_tests, or an explicit "
                    "accepted_without_validation with the user's rationale)"
                )
            elif validation_problem is not None:
                strict_violations.append(f"validation disposition: {validation_problem}")
        return MigrationRunFinalization(
            run_id=config.run_id,
            manifest_path=paths.manifest_path,
            report_path=paths.report_path,
            migration_complexity=plan.migration_complexity,
            unresolved_blockers=[blocker.rendered for blocker in plan.blockers],
            resolved_blockers=resolved,
            stale_decisions=stale,
            adapted_prompts=adapted_prompts,
            adapted_files=adapted_files,
            reviewed_unchanged=reviewed_unchanged,
            undecided_changes=undecided_changes,
            coverage_gaps=gaps,
            consistency_findings=[finding.rendered for finding in consistency],
            unconfirmed_unaffected=list(tasks.unaffected_files),
            validation_disposition=(
                validation_disposition.method if validation_disposition else None
            ),
            strict_violations=strict_violations,
            contract_test_path=contract_test_path,
            unknowns_open=[item.rendered for item in plan.unknowns if item.is_open],
            unknowns_closed_by_action=[
                f"{item.subject} [{item.id}]: {item.closed_reason}"
                for item in plan.unknowns
                if item.status == "closed_by_action"
            ],
            unknowns_closed_by_scan=[
                f"{item.subject} [{item.id}]: {item.closed_reason}"
                for item in plan.unknowns
                if item.status == "closed_by_scan"
            ],
            probe_paths=[str(workspace / item) for item in probe_paths],
            message=(
                (
                    f"STRICT MODE: {len(strict_violations)} unmet requirement(s) — this "
                    "run is NOT deliverable until they are resolved. "
                    if strict_violations
                    else ""
                )
                + "Migration run finalized. Review output/migration-report.md; "
                + (
                    f"{len(gaps)} affected file(s) still lack an adaptation deliverable; "
                    if gaps
                    else "every affected file has an adaptation deliverable; "
                )
                + (
                    f"{reviewed_unchanged} deliverable(s) were reviewed and needed no change; "
                    if reviewed_unchanged
                    else ""
                )
                + (
                    f"{len(tasks.unaffected_files)} unaffected file(s) still await a "
                    "confirm_unaffected entry; "
                    if tasks.unaffected_files
                    else ""
                )
                + (
                    f"{len(consistency)} cross-surface consistency finding(s) need "
                    "review (see the report's consistency section); "
                    if consistency
                    else ""
                )
                + (
                    f"{len(undecided_changes)} deliverable(s) have annotated changes "
                    "awaiting review decisions (get_change_review / "
                    "record_change_decision); "
                    if undecided_changes
                    else ""
                )
                + blocker_note
                + (
                    f" {sum(item.is_open for item in plan.unknowns)} unknown(s) remain open "
                    "(each with its action in the report)."
                    if any(item.is_open for item in plan.unknowns)
                    else ""
                )
            ),
        )
