"""Shared application service layer used by CLI and MCP transports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

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
    ResearchConsensus,
    ResearchScope,
    ResearchTopic,
    SourcePolicy,
    artifact_sha256,
    build_research_consensus,
    topic_for_field_path,
)
from llm_migrate.core.artifacts import write_proposal_artifacts
from llm_migrate.core.blockers import (
    RESOLUTION_GUIDANCE,
    BlockerDecisionResult,
    BlockerResolutionSet,
    apply_decisions,
    build_blocker_resolutions,
    load_decision_log,
    render_applied_decision,
    save_decision_log,
    upsert_decision,
)
from llm_migrate.core.comparison import compare_models
from llm_migrate.core.evaluation import (
    CustomEvaluator,
    EvaluationExecutor,
    analyze_regressions,
    compare_outputs,
    evaluation_artifact_as_yaml,
    generate_eval_suite,
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
    PromptMigrationSpec,
    PromptValidationResult,
    RecommendationConstraints,
    RecommendationResult,
    ResolutionKind,
    ResolvedModel,
    ValidationIssue,
    ValidationLevel,
    migration_blocker,
)
from llm_migrate.core.orchestration import (
    AgentRunner,
    WorkflowOutcome,
    run_agent_research_workflow,
)
from llm_migrate.core.planning import (
    generate_application_migration_plan,
    generate_migration_report,
    migration_manifest_as_yaml,
)
from llm_migrate.core.prompt_documents import (
    PromptDocumentError,
    extract_prompt_components,
    parse_structured_document,
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
from llm_migrate.core.session import (
    SessionRegistryManifest,
    build_session_registry,
    load_session_overlay,
    manifest_summary_lines,
    registry_content_sha256,
)
from llm_migrate.core.workspace import (
    AdaptationTaskList,
    FileSubmissionResult,
    MigrationRunConfig,
    MigrationRunFinalization,
    MigrationRunStart,
    PromptSubmissionResult,
    ResearchNeed,
    coverage_gaps,
    default_run_dir,
    default_run_id,
    derive_adaptation_tasks,
    load_adaptation_log,
    load_run_config,
    read_original_prompt,
    render_adaptation_section,
    run_paths,
    sanitize_run_id,
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

    def scan_application(
        self,
        root: Path | str,
        *,
        prompt_sources: Sequence[str] | None = None,
    ) -> ApplicationAnalysis:
        return scan_application(root, prompt_sources=prompt_sources)

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
        """
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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(request.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
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
    ) -> MigrationRunStart:
        """Resolve both models registry-first and prepare one run workspace.

        Nothing is written until both models resolve unambiguously; unresolved
        identifiers return candidates for explicit user confirmation instead.
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
            created_on=as_of,
            prompt_sources=list(prompt_sources or []),
        )
        write_run_config(run_dir, config)
        next_steps: list[str] = []
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
                "evidence-linked guidance and call submit_adapted_prompt; for every "
                "file task, write the complete adapted file and call "
                "submit_adapted_file, or pass unchanged=true when the file needs no "
                "change for the target model.",
                "Call finalize_migration(run_dir) to write migration-manifest.yaml, "
                "migration-report.md, and the adaptation change log under output/; it "
                "reports any remaining coverage gaps.",
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
            warnings=[*source_match.notes, *target_match.notes, *analysis.warnings],
            next_steps=next_steps,
        )

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
    ) -> MigrationPlan:
        """Plan a run over canonical knowledge plus its session overlay when built.

        Recorded blocker decisions are re-applied on every regeneration:
        redesign/accept decisions suppress exactly the live blocker they name
        (injecting the required redesign task), and decisions that no longer
        match a live blocker are reported as stale, never silently applied.
        """
        service, session_lines = self._run_service(run_dir, now=now)
        plan = service.generate_migration_plan(
            config.application_root,
            config.source.model,
            config.target.model,
            source_platform=config.source.platform,
            target_platform=config.target.platform,
            source_endpoint=config.source.endpoint,
            target_endpoint=config.target.endpoint,
            prompt_sources=config.prompt_sources or None,
        )
        if session_lines:
            plan = plan.model_copy(update={"warnings": [*plan.warnings, *session_lines]})
        return apply_decisions(plan, load_decision_log(run_dir, config.run_id), config)

    def list_adaptation_tasks(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> AdaptationTaskList:
        """Per-file adaptation worklist derived from the run's migration plan."""
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        plan = self._plan_for_run(config, workspace, now=now)
        return derive_adaptation_tasks(config, plan, workspace)

    def get_blocker_resolutions(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> BlockerResolutionSet:
        """Per-blocker questions with registry-backed options for user decisions."""
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        service, _ = self._run_service(workspace, now=now)
        plan = self._plan_for_run(config, workspace, now=now)
        analysis = self.scan_application(
            config.application_root, prompt_sources=config.prompt_sources or None
        )
        return BlockerResolutionSet(
            run_id=config.run_id,
            resolutions=build_blocker_resolutions(
                service.registry, config, plan, analysis, str(workspace)
            ),
            decisions=plan.decisions,
            guidance=list(RESOLUTION_GUIDANCE),
        )

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
        config = load_run_config(workspace)
        resolutions = self.get_blocker_resolutions(workspace, now=now)
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
        if option.kind is ResolutionKind.ACCEPT_WITH_RATIONALE and not rationale.strip():
            return BlockerDecisionResult(
                accepted=False,
                problems=[
                    "an accept decision requires the user's own free-text rationale; "
                    "record their words, not a placeholder"
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
        service, _ = self._run_service(workspace, now=now)
        config_updated = False
        if option.kind in {ResolutionKind.RETARGET, ResolutionKind.CORRECTION}:
            side = "target" if option.kind is ResolutionKind.RETARGET else "source"
            change = option.target_change if side == "target" else option.source_change
            assert change is not None
            identity = config.target if side == "target" else config.source
            try:
                if change.model and change.platform is None:
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
            config = config.model_copy(
                update={
                    side: new_identity,
                    f"{side}_model_id": resolved.platform.model_id,
                }
            )
            write_run_config(workspace, config)
            config_updated = True
        log = upsert_decision(load_decision_log(workspace, config.run_id), decision)
        save_decision_log(workspace, log)
        plan = self._plan_for_run(config, workspace, now=now)
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
        submitted_on: date | None = None,
    ) -> PromptSubmissionResult:
        """Validate and persist one adapted prompt beneath the run's output/prompts/."""
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        if unchanged and adapted_prompt.strip():
            raise ValueError(
                "unchanged=true cannot be combined with adapted content; omit the "
                "content or drop unchanged"
            )
        if unchanged:
            original = read_original_prompt(config, source_path)
            if original is not None:
                adapted_prompt = original
        validation = self._validate_prompt_submission(config, source_path, adapted_prompt)
        return workspace_submit_adapted_prompt(
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
        )

    def submit_adapted_file(
        self,
        run_dir: Path | str,
        source_path: str,
        adapted_content: str,
        rationale: str,
        changes: list[str],
        *,
        new_file: bool = False,
        unchanged: bool = False,
        submitted_on: date | None = None,
    ) -> FileSubmissionResult:
        """Check and persist one adapted application file beneath output/files/."""
        if unchanged and adapted_content.strip():
            raise ValueError(
                "unchanged=true cannot be combined with adapted content; omit the "
                "content or drop unchanged"
            )
        workspace = Path(run_dir)
        return workspace_submit_adapted_file(
            workspace,
            load_run_config(workspace),
            source_path,
            adapted_content,
            rationale,
            changes,
            submitted_on or date.today(),
            new_file=new_file,
            unchanged=unchanged,
        )

    def finalize_migration_run(
        self,
        run_dir: Path | str,
        *,
        now: datetime | None = None,
    ) -> MigrationRunFinalization:
        """Write the manifest, the report with per-file changes/rationale, and gaps."""
        workspace = Path(run_dir)
        config = load_run_config(workspace)
        plan = self._plan_for_run(config, workspace, now=now)
        log = load_adaptation_log(workspace, config.run_id)
        paths = run_paths(workspace)
        Path(paths.output_dir).mkdir(parents=True, exist_ok=True)
        Path(paths.manifest_path).write_text(
            self.migration_manifest_as_yaml(plan), encoding="utf-8"
        )
        tasks = derive_adaptation_tasks(config, plan, workspace, log)
        gaps = coverage_gaps(tasks)
        report = self.migration_report(plan)
        report = report.rstrip("\n") + "\n\n" + render_adaptation_section(log, gaps) + "\n"
        Path(paths.report_path).write_text(report, encoding="utf-8")
        adapted_prompts = sum(entry.kind == "prompt" for entry in log.entries)
        adapted_files = sum(entry.kind == "file" for entry in log.entries)
        resolved = [
            render_applied_decision(item) for item in plan.decisions if item.status == "applied"
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
            coverage_gaps=gaps,
            message=(
                "Migration run finalized. Review output/migration-report.md; "
                + (
                    f"{len(gaps)} affected file(s) still lack an adaptation deliverable; "
                    if gaps
                    else "every affected file has an adaptation deliverable; "
                )
                + blocker_note
            ),
        )
