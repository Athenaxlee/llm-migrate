"""Shared application service layer used by CLI and MCP transports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

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
    ComparisonSeverity,
    InvocationAnalysis,
    InvocationMigrationSpec,
    LivePricingResult,
    MigrationCostEstimate,
    MigrationPlan,
    MigrationWorkload,
    ModelComparison,
    ModelDifference,
    ModelLifecycleCheck,
    ModelProfile,
    PricingMatchStatus,
    PromptAnalysis,
    PromptMigrationSpec,
    PromptValidationResult,
    RecommendationConstraints,
    RecommendationResult,
    ResolvedModel,
    ValidationIssue,
    ValidationLevel,
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
from llm_migrate.core.proposals import propose_registry_update
from llm_migrate.core.recommendation import recommend_models
from llm_migrate.core.registry import ModelRegistry, RegistryError
from llm_migrate.core.resolver import AmbiguousModelError, ModelNotFoundError, resolve_model
from llm_migrate.core.session import (
    SessionRegistryManifest,
    build_session_registry,
    load_session_overlay,
    registry_content_sha256,
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

    def scan_application(self, root: Path | str) -> ApplicationAnalysis:
        return scan_application(root)

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
    ) -> PromptValidationResult:
        resolution = self.resolve_model(target)
        ambiguous_platform = False
        if target_platform:
            try:
                resolution = self.resolve_model(target, target_platform)
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
    ) -> InvocationMigrationSpec:
        analysis = (
            application
            if isinstance(application, ApplicationAnalysis)
            else self.scan_application(application)
        )
        source_resolution = self.resolve_model(source, source_platform)
        consistency_warnings: list[str] = []
        consistency_blockers: list[str] = []
        detected_canonical: set[str] = set()
        for identifier in sorted(analysis.requirements.source_models):
            try:
                detected_canonical.add(self.resolve_model(identifier).canonical_name)
            except ModelNotFoundError:
                consistency_warnings.append(
                    f"Detected model identifier {identifier!r} is not in the registry."
                )
        if detected_canonical and source_resolution.canonical_name not in detected_canonical:
            consistency_blockers.append(
                "Declared source model does not match detected model(s): "
                + ", ".join(sorted(detected_canonical))
                + "."
            )
        elif len(detected_canonical) > 1:
            consistency_warnings.append(
                "Multiple source models were detected; this preparation covers only "
                f"{source_resolution.canonical_name}."
            )
        detected_providers = analysis.requirements.source_providers
        if detected_providers and source_resolution.identity.provider not in detected_providers:
            consistency_blockers.append(
                "Declared source provider does not match detected provider(s): "
                + ", ".join(sorted(detected_providers))
                + "."
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
                "Declared source platform does not match detected platform(s): "
                + ", ".join(sorted(analysis.requirements.source_platforms))
                + "."
            )
        elif len(analysis.requirements.source_platforms) > 1:
            consistency_warnings.append(
                "Multiple source platforms were detected; review each invocation separately."
            )
        return prepare_invocation_migration(
            analysis,
            source_resolution,
            self.resolve_model(target, target_platform),
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
    ) -> MigrationPlan:
        analysis = (
            application
            if isinstance(application, ApplicationAnalysis)
            else self.scan_application(application)
        )
        source_resolution = self.resolve_model(source, source_platform)
        target_resolution = self.resolve_model(target, target_platform)
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
        )
        invocation = self.prepare_invocation_migration(
            analysis,
            source_resolution.canonical_name,
            target_resolution.canonical_name,
            source_platform=source_representation.platform,
            target_platform=target_representation.platform,
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
    ) -> str:
        plan = self.generate_migration_plan(
            application,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
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
