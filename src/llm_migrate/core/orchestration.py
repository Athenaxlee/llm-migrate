"""V1.1 resumable, fail-closed orchestration of bounded agent research.

The agent host (Claude Code, Codex, or a headless `AgentRunner`) supplies and
pays for every agent call. This module owns everything deterministic: stage
sequencing, retries, budgets, artifact persistence, hash links, and the gates
between stages. Repository and webpage content flowing through artifacts is
data; nothing an agent returns can widen limits, change stages, or gain write
authority over the canonical registry.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from llm_migrate.core.agent_research import (
    AGENT_ID_PATTERN,
    SHA256_PATTERN,
    ConsensusError,
    EvidenceReview,
    ExecutionLimits,
    MigrationResearchRequest,
    ModelEndpointIdentity,
    ResearchConsensus,
    ResearchScope,
    ResearchTopic,
    SessionSuitability,
    SourcePolicy,
    artifact_sha256,
    build_research_consensus,
    compile_migration_knowledge_candidate,
)
from llm_migrate.core.knowledge import MigrationKnowledge, ResearchResult
from llm_migrate.core.models import ModelProfile, StrictModel
from llm_migrate.core.proposals import propose_registry_update
from llm_migrate.core.registry import ModelRegistry
from llm_migrate.core.session import (
    SessionRegistryManifest,
    build_session_manifest,
    write_session_overlay,
)


class AgentRole(StrEnum):
    SOURCE_RESEARCHER = "source_researcher"
    TARGET_RESEARCHER = "target_researcher"
    MIGRATION_RESEARCHER = "migration_researcher"
    EVIDENCE_REVIEWER = "evidence_reviewer"
    DISPUTE_ARBITER = "dispute_arbiter"


class ResearchAssignment(StrictModel):
    """Everything a research agent receives; never the full repository."""

    schema_version: Literal["1"] = "1"
    role: AgentRole
    scope: ResearchScope
    subject: str = Field(min_length=1)
    source: ModelEndpointIdentity
    target: ModelEndpointIdentity
    canonical_model_candidate: str | None = None
    topics: list[ResearchTopic] = Field(min_length=1)
    as_of: date
    application_requirements_sha256: str = Field(pattern=SHA256_PATTERN)
    source_policy: SourcePolicy
    limits: ExecutionLimits


class ReviewAssignment(StrictModel):
    """What the independent reviewer receives: the artifact, not the agent."""

    schema_version: Literal["1"] = "1"
    role: AgentRole = AgentRole.EVIDENCE_REVIEWER
    subject: str = Field(min_length=1)
    researcher_id: str = Field(min_length=1, pattern=AGENT_ID_PATTERN)
    research: ResearchResult
    research_sha256: str = Field(pattern=SHA256_PATTERN)
    as_of: date
    source_policy: SourcePolicy
    limits: ExecutionLimits


class AgentUsage(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class AgentRunner(Protocol):
    """Host-supplied agent execution. The project owns no credentials."""

    def run_research(
        self, assignment: ResearchAssignment
    ) -> tuple[str, ResearchResult, AgentUsage]:
        """Return (agent_id, result, usage) for one bounded research assignment."""
        ...

    def run_review(self, assignment: ReviewAssignment) -> tuple[str, EvidenceReview, AgentUsage]:
        """Return (agent_id, review, usage) for one independent review assignment."""
        ...


class NullAgentRunner:
    """Refuses agent work; usable when every artifact is already on disk.

    Agent hosts that drive stages themselves write `research/<scope>.yaml` and
    `review/<scope>.yaml` into the run workspace and then finalize with this
    runner: completed artifacts resume, and any stage that would still need an
    agent fails closed instead of silently doing network or inference work.
    """

    def run_research(
        self, assignment: ResearchAssignment
    ) -> tuple[str, ResearchResult, AgentUsage]:
        raise OrchestrationError(
            f"no agent runtime is configured; write the research artifact for "
            f"{assignment.subject!r} into the run workspace first"
        )

    def run_review(self, assignment: ReviewAssignment) -> tuple[str, EvidenceReview, AgentUsage]:
        raise OrchestrationError(
            f"no agent runtime is configured; write the evidence review for "
            f"{assignment.subject!r} into the run workspace first"
        )


class StageStatus(StrEnum):
    COMPLETED = "completed"
    RESUMED = "resumed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageRecord(StrictModel):
    name: str = Field(min_length=1)
    status: StageStatus
    artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    agent_id: str | None = None
    attempts: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    failure: str | None = None


class StoppingReason(StrEnum):
    COMPLETED = "completed"
    STAGE_FAILED = "stage_failed"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"


class OrchestrationRun(StrictModel):
    """Resumable run record: hashes and usage, never secrets or reasoning."""

    schema_version: Literal["1"] = "1"
    run_id: str
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    stages: list[StageRecord] = Field(default_factory=list)
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    stopping_reason: StoppingReason
    warnings: list[str] = Field(default_factory=list)


class WorkflowOutcome(StrictModel):
    run: OrchestrationRun
    manifest: SessionRegistryManifest | None = None
    consensuses: list[ResearchConsensus] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class OrchestrationError(ValueError):
    """The workflow cannot run or resume as requested."""


_DEFAULT_OVERLAY_TTL = timedelta(days=7)


def _write_yaml(path: Path, artifact: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(artifact.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_yaml(path: Path, schema: type[ModelT]) -> ModelT | None:
    if not path.is_file():
        return None
    try:
        return schema.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise OrchestrationError(f"corrupt run artifact {path}: {exc}") from exc


def _scope_subject(request: MigrationResearchRequest, scope: ResearchScope) -> str:
    if scope is ResearchScope.SOURCE:
        return request.source.model
    if scope is ResearchScope.TARGET:
        return request.target.model
    return f"{request.source.model} -> {request.target.model}"


def _filtered_to_accepted(research: ResearchResult, consensus: ResearchConsensus) -> ResearchResult:
    """Keep only claims that survived independent review before compilation."""
    accepted = set(consensus.accepted_claim_ids)
    claims = [claim for claim in research.claims if claim.id in accepted]
    used_sources = {source_id for claim in claims for source_id in claim.sources}
    observation_sources = {
        source_id for observation in research.observations for source_id in observation.sources
    }
    return ResearchResult(
        subject=research.subject,
        canonical_model_candidate=research.canonical_model_candidate,
        research_date=research.research_date,
        claims=claims,
        sources=[
            source
            for source in research.sources
            if source.id in used_sources or source.id in observation_sources
        ],
        observations=research.observations,
        conflicts=[],
        unresolved_questions=research.unresolved_questions,
        warnings=research.warnings,
    )


class _WorkflowState:
    def __init__(self, run_dir: Path, request: MigrationResearchRequest) -> None:
        self.run_dir = run_dir
        self.request = request
        self.stages: list[StageRecord] = []
        self.total_input = 0
        self.total_output = 0
        self.blockers: list[str] = []
        self.warnings: list[str] = []
        prior = _read_yaml(run_dir / "run.yaml", OrchestrationRun)
        if prior is not None:
            if prior.request_sha256 != artifact_sha256(request):
                raise OrchestrationError(
                    "existing run workspace was created for a different research request"
                )
            self.total_input = prior.total_input_tokens
            self.total_output = prior.total_output_tokens

    def budget_exhausted(self) -> bool:
        return (self.total_input + self.total_output) >= self.request.limits.max_total_tokens

    def record(self, record: StageRecord) -> None:
        self.stages.append(record)

    def add_usage(self, usage: AgentUsage) -> None:
        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens

    def snapshot(self, stopping_reason: StoppingReason) -> OrchestrationRun:
        run = OrchestrationRun(
            run_id=self.request.run_id,
            request_sha256=artifact_sha256(self.request),
            stages=self.stages,
            total_input_tokens=self.total_input,
            total_output_tokens=self.total_output,
            stopping_reason=stopping_reason,
            warnings=self.warnings,
        )
        _write_yaml(self.run_dir / "run.yaml", run)
        return run


def run_agent_research_workflow(
    base_registry: ModelRegistry,
    request: MigrationResearchRequest,
    runner: AgentRunner,
    workspace: Path,
    *,
    now: datetime,
    base_registry_sha256: str,
    overlay_ttl: timedelta = _DEFAULT_OVERLAY_TTL,
    shadow_canonical: list[str] | None = None,
) -> WorkflowOutcome:
    """Drive research → review → consensus → overlay, resumably and fail-closed.

    Stage artifacts persist under `<workspace>/<run-id>/`; rerunning with the
    same request reuses completed artifacts instead of repeating agent work.
    """
    run_dir = Path(workspace) / request.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.yaml"
    existing_request = _read_yaml(request_path, MigrationResearchRequest)
    if existing_request is None:
        _write_yaml(request_path, request)
    elif artifact_sha256(existing_request) != artifact_sha256(request):
        raise OrchestrationError(
            "existing run workspace was created for a different research request"
        )

    state = _WorkflowState(run_dir, request)
    scopes = [
        scope
        for scope in (ResearchScope.SOURCE, ResearchScope.TARGET, ResearchScope.PAIR)
        if scope in request.scopes
    ]
    role_by_scope = {
        ResearchScope.SOURCE: AgentRole.SOURCE_RESEARCHER,
        ResearchScope.TARGET: AgentRole.TARGET_RESEARCHER,
        ResearchScope.PAIR: AgentRole.MIGRATION_RESEARCHER,
    }

    research_by_scope: dict[ResearchScope, ResearchResult] = {}
    researcher_by_scope: dict[ResearchScope, str] = {}
    review_by_scope: dict[ResearchScope, EvidenceReview] = {}
    stopping = StoppingReason.COMPLETED

    def fail(record: StageRecord, reason: StoppingReason) -> WorkflowOutcome:
        state.record(record)
        run = state.snapshot(reason)
        return WorkflowOutcome(
            run=run,
            manifest=None,
            consensuses=[],
            blockers=[*state.blockers, record.failure or reason.value],
        )

    for phase in ("research", "review"):
        for scope in scopes:
            stage_name = f"{phase}_{scope.value}"
            artifact_path = run_dir / phase / f"{scope.value}.yaml"
            if state.budget_exhausted():
                state.record(
                    StageRecord(
                        name=stage_name,
                        status=StageStatus.SKIPPED,
                        failure="token budget exhausted before this stage",
                    )
                )
                stopping = StoppingReason.TOKEN_BUDGET_EXHAUSTED
                continue
            if stopping is not StoppingReason.COMPLETED:
                state.record(
                    StageRecord(
                        name=stage_name,
                        status=StageStatus.SKIPPED,
                        failure="an earlier stage failed",
                    )
                )
                continue
            if phase == "review" and scope not in research_by_scope:
                state.record(
                    StageRecord(
                        name=stage_name,
                        status=StageStatus.SKIPPED,
                        failure="no research artifact to review",
                    )
                )
                continue

            existing: ResearchResult | EvidenceReview | None = (
                _read_yaml(artifact_path, ResearchResult)
                if phase == "research"
                else _read_yaml(artifact_path, EvidenceReview)
            )
            if existing is not None:
                agent_path = artifact_path.with_suffix(".agent")
                agent_id = (
                    agent_path.read_text(encoding="utf-8").strip() if agent_path.is_file() else None
                )
                if isinstance(existing, ResearchResult):
                    research_by_scope[scope] = existing
                    if agent_id:
                        researcher_by_scope[scope] = agent_id
                else:
                    review_by_scope[scope] = existing
                state.record(
                    StageRecord(
                        name=stage_name,
                        status=StageStatus.RESUMED,
                        artifact_sha256=artifact_sha256(existing),
                        agent_id=agent_id,
                    )
                )
                continue

            attempts = 0
            last_error: str | None = None
            produced: ResearchResult | EvidenceReview | None = None
            produced_agent: str | None = None
            usage_total = AgentUsage()
            while attempts < request.limits.max_attempts_per_stage and produced is None:
                attempts += 1
                try:
                    if phase == "research":
                        assignment = ResearchAssignment(
                            role=role_by_scope[scope],
                            scope=scope,
                            subject=_scope_subject(request, scope),
                            source=request.source,
                            target=request.target,
                            canonical_model_candidate=(
                                request.source.model
                                if scope is ResearchScope.SOURCE
                                else request.target.model
                                if scope is ResearchScope.TARGET
                                else None
                            ),
                            topics=request.topics,
                            as_of=request.as_of,
                            application_requirements_sha256=(
                                request.application_requirements_sha256
                            ),
                            source_policy=request.source_policy,
                            limits=request.limits,
                        )
                        agent_id, result, usage = runner.run_research(assignment)
                        produced, produced_agent = result, agent_id
                    else:
                        research = research_by_scope[scope]
                        review_assignment = ReviewAssignment(
                            subject=research.subject,
                            researcher_id=researcher_by_scope.get(scope, "unknown-researcher"),
                            research=research,
                            research_sha256=artifact_sha256(research),
                            as_of=request.as_of,
                            source_policy=request.source_policy,
                            limits=request.limits,
                        )
                        agent_id, review, usage = runner.run_review(review_assignment)
                        if review.researcher_id != review_assignment.researcher_id:
                            raise OrchestrationError("review does not name the assigned researcher")
                        produced, produced_agent = review, agent_id
                    usage_total = AgentUsage(
                        input_tokens=usage_total.input_tokens + usage.input_tokens,
                        output_tokens=usage_total.output_tokens + usage.output_tokens,
                    )
                except (ValidationError, ValueError) as exc:
                    last_error = str(exc)

            state.add_usage(usage_total)
            if produced is None:
                record = StageRecord(
                    name=stage_name,
                    status=StageStatus.FAILED,
                    attempts=attempts,
                    input_tokens=usage_total.input_tokens,
                    output_tokens=usage_total.output_tokens,
                    failure=last_error or "agent produced no artifact",
                )
                return fail(record, StoppingReason.STAGE_FAILED)
            _write_yaml(artifact_path, produced)
            if produced_agent:
                artifact_path.with_suffix(".agent").write_text(produced_agent, encoding="utf-8")
            if isinstance(produced, ResearchResult):
                research_by_scope[scope] = produced
                if produced_agent:
                    researcher_by_scope[scope] = produced_agent
            else:
                review_by_scope[scope] = produced
            state.record(
                StageRecord(
                    name=stage_name,
                    status=StageStatus.COMPLETED,
                    artifact_sha256=artifact_sha256(produced),
                    agent_id=produced_agent,
                    attempts=attempts,
                    input_tokens=usage_total.input_tokens,
                    output_tokens=usage_total.output_tokens,
                )
            )

    if stopping is not StoppingReason.COMPLETED or any(
        record.status is StageStatus.FAILED for record in state.stages
    ):
        run = state.snapshot(stopping)
        return WorkflowOutcome(
            run=run,
            manifest=None,
            consensuses=[],
            blockers=[*state.blockers, "agent stages did not complete; failing closed"],
        )

    consensuses: list[ResearchConsensus] = []
    for scope in scopes:
        stage_name = f"consensus_{scope.value}"
        research = research_by_scope[scope]
        review = review_by_scope[scope]
        try:
            consensus = build_research_consensus(
                research,
                review,
                request.topics,
                request.source_policy,
            )
        except ConsensusError as exc:
            record = StageRecord(
                name=stage_name,
                status=StageStatus.FAILED,
                failure=str(exc),
            )
            return fail(record, StoppingReason.STAGE_FAILED)
        _write_yaml(run_dir / "consensus" / f"{scope.value}.yaml", consensus)
        consensuses.append(consensus)
        state.record(
            StageRecord(
                name=stage_name,
                status=StageStatus.COMPLETED,
                artifact_sha256=artifact_sha256(consensus),
            )
        )
        state.blockers.extend(consensus.blockers)
        state.warnings.extend(consensus.warnings)

    candidates: list[ModelProfile] = []
    knowledge: list[MigrationKnowledge] = []
    for scope, consensus in zip(scopes, consensuses, strict=True):
        research = research_by_scope[scope]
        if consensus.suitability is not SessionSuitability.SUITABLE:
            continue
        if scope is ResearchScope.PAIR:
            candidate_knowledge = compile_migration_knowledge_candidate(
                f"session-{request.run_id}-migration",
                request.source.model,
                request.target.model,
                research,
                consensus,
            )
            if candidate_knowledge is not None:
                knowledge.append(candidate_knowledge)
            continue
        filtered = _filtered_to_accepted(research, consensus)
        proposal = propose_registry_update(filtered, base_registry)
        if proposal.candidate_model is None:
            state.blockers.append(
                f"session candidate for {research.subject!r} failed validation: "
                + "; ".join(proposal.candidate_errors)
            )
            continue
        candidates.append(ModelProfile.model_validate(proposal.candidate_model))

    manifest = build_session_manifest(
        run_id=request.run_id,
        created_at=now,
        expires_at=now + overlay_ttl,
        base_registry_sha256=base_registry_sha256,
        request_sha256=artifact_sha256(request),
        consensuses=consensuses,
        candidate_profiles=candidates,
        migration_knowledge=knowledge,
        shadowed_canonical=shadow_canonical,
    )
    write_session_overlay(run_dir, manifest, candidates, knowledge)
    state.record(
        StageRecord(
            name="session_registry",
            status=StageStatus.COMPLETED,
            artifact_sha256=artifact_sha256(manifest),
        )
    )
    run = state.snapshot(StoppingReason.COMPLETED)
    return WorkflowOutcome(
        run=run,
        manifest=manifest,
        consensuses=consensuses,
        blockers=sorted(set(state.blockers)),
    )
