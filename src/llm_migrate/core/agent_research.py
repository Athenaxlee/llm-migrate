"""V1.1 typed contracts and deterministic rules for bounded agent research.

Agent hosts produce `ResearchResult` and `EvidenceReview` artifacts; everything
that decides what those artifacts mean — acceptance, conflicts, coverage,
trust, and suitability — is deterministic service logic in this module. Agent
agreement is never treated as evidence, and no researcher may review its own
result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from llm_migrate.core.knowledge import (
    EvidenceClaim,
    EvidenceKind,
    MigrationKnowledge,
    MigrationKnowledgeItem,
    ResearchResult,
)
from llm_migrate.core.models import StrictModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
AGENT_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


def artifact_sha256(artifact: BaseModel) -> str:
    """Stable content hash linking every stage artifact to the next."""
    payload = json.dumps(
        artifact.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResearchTopic(StrEnum):
    PRICING = "pricing"
    LIFECYCLE = "lifecycle"
    AVAILABILITY = "availability"
    CAPABILITIES = "capabilities"
    PARAMETERS = "parameters"
    PROMPTING_GUIDANCE = "prompting_guidance"
    MIGRATION_BEHAVIOR = "migration_behavior"


def topic_for_field_path(field_path: str) -> ResearchTopic | None:
    """Map a claim's field path onto the requested-topic vocabulary."""
    if field_path.startswith("pricing"):
        return ResearchTopic.PRICING
    if field_path.startswith("lifecycle"):
        return ResearchTopic.LIFECYCLE
    if field_path.startswith("platforms"):
        return ResearchTopic.AVAILABILITY
    if field_path.startswith("capabilities"):
        return ResearchTopic.CAPABILITIES
    if field_path.startswith("parameters"):
        return ResearchTopic.PARAMETERS
    if field_path.startswith("prompt_guidance"):
        return ResearchTopic.PROMPTING_GUIDANCE
    if field_path.startswith("migration_behavior"):
        return ResearchTopic.MIGRATION_BEHAVIOR
    return None


HIGH_IMPACT_TOPICS = frozenset(
    {
        ResearchTopic.PRICING,
        ResearchTopic.LIFECYCLE,
        ResearchTopic.AVAILABILITY,
        ResearchTopic.CAPABILITIES,
        ResearchTopic.MIGRATION_BEHAVIOR,
    }
)


class ModelEndpointIdentity(StrictModel):
    """Exact identity of one side of a migration, as the user requested it."""

    provider: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    model: str = Field(min_length=1)
    endpoint: str | None = None


class SourcePolicy(StrictModel):
    """Which sources may establish session facts. Fails closed."""

    maximum_fact_authority_tier: int = Field(default=3, ge=1, le=6)
    allow_community_observations: bool = True


class ExecutionLimits(StrictModel):
    """Bounded agent execution. Exhausted limits stop the run; they never widen."""

    max_research_agents: int = Field(default=3, ge=1, le=16)
    max_review_agents: int = Field(default=3, ge=1, le=16)
    max_sources_per_topic: int = Field(default=5, ge=1, le=25)
    max_attempts_per_stage: int = Field(default=2, ge=1, le=5)
    max_concurrency: int = Field(default=2, ge=1, le=8)
    max_total_tokens: int = Field(default=500_000, ge=1)
    max_wall_clock_seconds: int = Field(default=1_800, ge=1)


class ResearchScope(StrEnum):
    SOURCE = "source"
    TARGET = "target"
    PAIR = "pair"


class MigrationResearchRequest(StrictModel):
    """Explicit, bounded assignment created before any research runs."""

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1, pattern=RUN_ID_PATTERN)
    source: ModelEndpointIdentity
    target: ModelEndpointIdentity
    application_requirements_sha256: str = Field(pattern=SHA256_PATTERN)
    topics: list[ResearchTopic] = Field(min_length=1)
    scopes: list[ResearchScope] = Field(
        default_factory=lambda: [ResearchScope.SOURCE, ResearchScope.TARGET, ResearchScope.PAIR]
    )
    as_of: date
    source_policy: SourcePolicy = SourcePolicy()
    limits: ExecutionLimits = ExecutionLimits()

    @model_validator(mode="after")
    def validate_topics(self) -> MigrationResearchRequest:
        if len(self.topics) != len(set(self.topics)):
            raise ValueError("research topics must be unique")
        if not self.scopes:
            raise ValueError("at least one research scope is required")
        if len(self.scopes) != len(set(self.scopes)):
            raise ValueError("research scopes must be unique")
        return self


class VerdictKind(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STALE = "stale"
    INCORRECTLY_SCOPED = "incorrectly_scoped"


class ClaimScope(StrictModel):
    provider: str = Field(min_length=1)
    platform: str | None = None


class ClaimVerdict(StrictModel):
    """One independent verdict for one research claim."""

    claim_id: str = Field(min_length=1)
    verdict: VerdictKind
    checked_source_ids: list[str] = Field(default_factory=list)
    scope: ClaimScope
    rationale: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_supported_requires_checked_sources(self) -> ClaimVerdict:
        if self.verdict is VerdictKind.SUPPORTED and not self.checked_source_ids:
            raise ValueError("a supported verdict must record independently checked sources")
        return self


class EvidenceReview(StrictModel):
    """Independent claim-level review of one `ResearchResult`."""

    schema_version: Literal["1"] = "1"
    reviewer_id: str = Field(min_length=1, pattern=AGENT_ID_PATTERN)
    researcher_id: str = Field(min_length=1, pattern=AGENT_ID_PATTERN)
    research_sha256: str = Field(pattern=SHA256_PATTERN)
    reviewed_at: date
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review(self) -> EvidenceReview:
        if self.reviewer_id == self.researcher_id:
            raise ValueError("an agent cannot review its own research result")
        claim_ids = [verdict.claim_id for verdict in self.verdicts]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("review verdict claim IDs must be unique")
        return self


class ResearchArtifactValidation(StrictModel):
    """Result of validating one scope's workspace research artifacts by path.

    Path-based validation reads the researcher's (and, when present, the
    reviewer's) YAML from the run workspace instead of requiring the full
    artifact to be resent through a transport payload.
    """

    schema_version: Literal["1"] = "1"
    run_id: str
    scope: str
    research_path: str = ""
    review_path: str = ""
    research_checked: bool = False
    review_checked: bool = False
    valid: bool
    problems: list[str] = Field(default_factory=list)
    message: str


class TopicCoverage(StrEnum):
    COVERED = "covered"
    MISSING = "missing"
    BLOCKED = "blocked"


class SessionSuitability(StrEnum):
    SUITABLE = "suitable"
    UNSUITABLE = "unsuitable"


class TrustLevel(StrEnum):
    CANONICAL_VERIFIED = "canonical_verified"
    SESSION_AGENT_REVIEWED = "session_agent_reviewed"
    SESSION_UNREVIEWED = "session_unreviewed"


class ClaimResolution(StrictModel):
    claim_id: str
    reason: str


class UnresolvedConflict(StrictModel):
    field_path: str
    claim_ids: list[str] = Field(min_length=1)
    statement: str
    high_impact: bool


class ResearchConsensus(StrictModel):
    """Deterministic combination of one research result and its review."""

    schema_version: Literal["1"] = "1"
    subject: str = Field(min_length=1)
    research_sha256: str = Field(pattern=SHA256_PATTERN)
    review_sha256: str = Field(pattern=SHA256_PATTERN)
    accepted_claim_ids: list[str] = Field(default_factory=list)
    rejected_claims: list[ClaimResolution] = Field(default_factory=list)
    unresolved_conflicts: list[UnresolvedConflict] = Field(default_factory=list)
    topic_coverage: dict[ResearchTopic, TopicCoverage] = Field(default_factory=dict)
    trust_level: TrustLevel
    suitability: SessionSuitability
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConsensusError(ValueError):
    """The research/review pair cannot be combined at all."""


def claim_meets_source_threshold(
    claim: EvidenceClaim,
    source_tiers: dict[str, int],
    policy: SourcePolicy,
) -> tuple[bool, str]:
    """Source-authority acceptance rule shared with the canonical proposal path."""
    if claim.kind is not EvidenceKind.FACT:
        return False, f"{claim.kind.value} cannot establish a session fact"
    if not claim.sources:
        return False, "claim has no supporting source"
    best_tier = min(source_tiers[source_id] for source_id in claim.sources)
    if best_tier > policy.maximum_fact_authority_tier:
        return False, (
            f"best source tier {best_tier} exceeds the policy maximum "
            f"{policy.maximum_fact_authority_tier}"
        )
    if claim.confidence.value == "authoritative" and best_tier <= 3:
        return True, "authoritative claim supported by a tier 1-3 source"
    if claim.confidence.value == "high" and best_tier <= 2:
        return True, "high-confidence claim supported by a tier 1-2 source"
    return False, "evidence does not meet the session fact threshold"


def build_research_consensus(
    research: ResearchResult,
    review: EvidenceReview,
    requested_topics: list[ResearchTopic],
    policy: SourcePolicy | None = None,
) -> ResearchConsensus:
    """Deterministically accept or reject every claim; never vote.

    Raises `ConsensusError` when the review does not actually cover this
    research artifact (wrong hash or self-review) — those pairs must not be
    silently combined.
    """
    policy = policy or SourcePolicy()
    research_hash = artifact_sha256(research)
    if review.research_sha256 != research_hash:
        raise ConsensusError(
            "evidence review does not reference this research result "
            f"(expected {research_hash}, review carries {review.research_sha256})"
        )

    source_tiers = {source.id: int(source.authority_tier.value) for source in research.sources}
    verdicts = {verdict.claim_id: verdict for verdict in review.verdicts}
    conflicted_claim_ids = {
        claim_id for conflict in research.conflicts for claim_id in conflict.claim_ids
    }

    accepted: list[str] = []
    rejected: list[ClaimResolution] = []
    unresolved: list[UnresolvedConflict] = []
    blockers: list[str] = []
    warnings: list[str] = list(review.warnings)
    covered_topics: set[ResearchTopic] = set()
    blocked_topics: set[ResearchTopic] = set()

    for conflict in sorted(research.conflicts, key=lambda item: item.id):
        field_paths = sorted(
            {claim.field_path for claim in research.claims if claim.id in set(conflict.claim_ids)}
        )
        field_path = field_paths[0] if field_paths else "unknown"
        topic = topic_for_field_path(field_path)
        high_impact = topic in HIGH_IMPACT_TOPICS if topic else True
        unresolved.append(
            UnresolvedConflict(
                field_path=field_path,
                claim_ids=sorted(conflict.claim_ids),
                statement=conflict.statement,
                high_impact=high_impact,
            )
        )
        if high_impact and topic is not None:
            blocked_topics.add(topic)

    for claim in sorted(research.claims, key=lambda item: item.id):
        verdict = verdicts.get(claim.id)
        if verdict is None:
            rejected.append(
                ClaimResolution(
                    claim_id=claim.id,
                    reason="the independent review recorded no verdict for this claim",
                )
            )
            blockers.append(f"claim {claim.id!r} has no independent review verdict")
            continue
        if claim.id in conflicted_claim_ids:
            rejected.append(
                ClaimResolution(
                    claim_id=claim.id,
                    reason="claim participates in an unresolved evidence conflict",
                )
            )
            continue
        if verdict.verdict is VerdictKind.CONTRADICTED:
            topic = topic_for_field_path(claim.field_path)
            high_impact = topic in HIGH_IMPACT_TOPICS if topic else True
            rejected.append(
                ClaimResolution(
                    claim_id=claim.id,
                    reason=f"independent review contradicted the claim: {verdict.rationale}",
                )
            )
            unresolved.append(
                UnresolvedConflict(
                    field_path=claim.field_path,
                    claim_ids=[claim.id],
                    statement=verdict.rationale,
                    high_impact=high_impact,
                )
            )
            if high_impact and topic is not None:
                blocked_topics.add(topic)
            continue
        if verdict.verdict is not VerdictKind.SUPPORTED:
            rejected.append(
                ClaimResolution(
                    claim_id=claim.id,
                    reason=f"independent review verdict: {verdict.verdict.value}",
                )
            )
            continue
        if not set(verdict.checked_source_ids) & set(claim.sources):
            rejected.append(
                ClaimResolution(
                    claim_id=claim.id,
                    reason="the reviewer did not independently check any cited source",
                )
            )
            continue
        supported, reason = claim_meets_source_threshold(claim, source_tiers, policy)
        if not supported:
            rejected.append(ClaimResolution(claim_id=claim.id, reason=reason))
            continue
        if verdict.blockers:
            rejected.append(
                ClaimResolution(
                    claim_id=claim.id,
                    reason=f"review blockers: {'; '.join(verdict.blockers)}",
                )
            )
            blockers.extend(verdict.blockers)
            continue
        accepted.append(claim.id)
        topic = topic_for_field_path(claim.field_path)
        if topic is not None:
            covered_topics.add(topic)
        warnings.extend(verdict.warnings)

    for verdict in review.verdicts:
        if verdict.claim_id not in {claim.id for claim in research.claims}:
            blockers.append(f"review verdict references unknown claim {verdict.claim_id!r}")

    coverage: dict[ResearchTopic, TopicCoverage] = {}
    for topic in requested_topics:
        if topic in blocked_topics:
            coverage[topic] = TopicCoverage.BLOCKED
        elif topic in covered_topics:
            coverage[topic] = TopicCoverage.COVERED
        else:
            coverage[topic] = TopicCoverage.MISSING
            warnings.append(f"requested topic {topic.value!r} has no accepted claims")

    for open_conflict in unresolved:
        if open_conflict.high_impact:
            blockers.append(
                "high-impact unresolved conflict on "
                f"{open_conflict.field_path!r} remains unknown and blocks session use"
            )

    suitable = not blockers and bool(accepted)
    if not accepted and not blockers:
        warnings.append("no claim survived independent review")
    trust = TrustLevel.SESSION_AGENT_REVIEWED if suitable else TrustLevel.SESSION_UNREVIEWED

    return ResearchConsensus(
        subject=research.subject,
        research_sha256=research_hash,
        review_sha256=artifact_sha256(review),
        accepted_claim_ids=sorted(accepted),
        rejected_claims=sorted(rejected, key=lambda item: item.claim_id),
        unresolved_conflicts=unresolved,
        topic_coverage=coverage,
        trust_level=trust,
        suitability=SessionSuitability.SUITABLE if suitable else SessionSuitability.UNSUITABLE,
        blockers=sorted(set(blockers)),
        warnings=warnings,
    )


class DisputeArbitration(StrictModel):
    """A conditional third opinion on one unresolved conflict.

    The arbiter may add independently sourced evidence for one side; it can
    never waive schema, source, scope, coverage, freshness, or conflict rules.
    """

    schema_version: Literal["1"] = "1"
    arbiter_id: str = Field(min_length=1, pattern=AGENT_ID_PATTERN)
    conflict_claim_ids: list[str] = Field(min_length=1)
    upheld_claim_id: str | None = None
    added_evidence: ResearchResult | None = None
    rationale: str = Field(min_length=1)


def arbitrate_conflict(
    consensus: ResearchConsensus,
    research: ResearchResult,
    review: EvidenceReview,
    arbitration: DisputeArbitration,
    policy: SourcePolicy | None = None,
) -> ResearchConsensus:
    """Resolve one conflict only when a third party adds qualifying evidence."""
    policy = policy or SourcePolicy()
    if arbitration.arbiter_id in {review.reviewer_id, review.researcher_id}:
        raise ConsensusError("the dispute arbiter must be independent of researcher and reviewer")
    matching = [
        conflict
        for conflict in consensus.unresolved_conflicts
        if set(arbitration.conflict_claim_ids) & set(conflict.claim_ids)
    ]
    if not matching:
        raise ConsensusError("arbitration references no unresolved conflict in this consensus")
    if arbitration.upheld_claim_id is None:
        return consensus
    claims = {claim.id: claim for claim in research.claims}
    upheld = claims.get(arbitration.upheld_claim_id)
    if upheld is None:
        raise ConsensusError("arbitration upholds a claim that does not exist")
    if arbitration.added_evidence is None:
        raise ConsensusError(
            "arbitration cannot uphold a claim without independently sourced evidence"
        )
    added_tiers = {
        source.id: int(source.authority_tier.value) for source in arbitration.added_evidence.sources
    }
    corroborating = [
        claim
        for claim in arbitration.added_evidence.claims
        if claim.field_path == upheld.field_path and claim.value == upheld.value
    ]
    qualifying = [
        claim
        for claim in corroborating
        if claim_meets_source_threshold(claim, added_tiers, policy)[0]
    ]
    if not qualifying:
        raise ConsensusError(
            "arbitration evidence does not independently meet the source threshold"
        )
    remaining = [
        conflict
        for conflict in consensus.unresolved_conflicts
        if not (set(arbitration.conflict_claim_ids) & set(conflict.claim_ids))
    ]
    accepted = sorted({*consensus.accepted_claim_ids, upheld.id})
    rejected = [item for item in consensus.rejected_claims if item.claim_id != upheld.id]
    blockers = [
        blocker
        for blocker in consensus.blockers
        if not blocker.startswith("high-impact unresolved conflict")
        or all(
            conflict.field_path not in blocker
            for conflict in consensus.unresolved_conflicts
            if set(arbitration.conflict_claim_ids) & set(conflict.claim_ids)
        )
    ]
    for conflict in remaining:
        if conflict.high_impact:
            blocker = (
                "high-impact unresolved conflict on "
                f"{conflict.field_path!r} remains unknown and blocks session use"
            )
            if blocker not in blockers:
                blockers.append(blocker)
    coverage = dict(consensus.topic_coverage)
    topic = topic_for_field_path(upheld.field_path)
    if topic is not None and topic in coverage:
        coverage[topic] = TopicCoverage.COVERED
    suitable = not blockers and bool(accepted)
    return consensus.model_copy(
        update={
            "accepted_claim_ids": accepted,
            "rejected_claims": rejected,
            "unresolved_conflicts": remaining,
            "topic_coverage": coverage,
            "trust_level": (
                TrustLevel.SESSION_AGENT_REVIEWED if suitable else TrustLevel.SESSION_UNREVIEWED
            ),
            "suitability": (
                SessionSuitability.SUITABLE if suitable else SessionSuitability.UNSUITABLE
            ),
            "blockers": sorted(set(blockers)),
            "warnings": [
                *consensus.warnings,
                f"conflict on {upheld.field_path!r} resolved by arbiter "
                f"{arbitration.arbiter_id!r}: {arbitration.rationale}",
            ],
        }
    )


def compile_migration_knowledge_candidate(
    knowledge_id: str,
    source_model: str,
    target_model: str,
    research: ResearchResult,
    consensus: ResearchConsensus,
) -> MigrationKnowledge | None:
    """Compile accepted pair-specific claims into a validated candidate.

    Two valid model profiles do not establish migration behavior between them;
    only accepted `migration_behavior.*` claims can. Returns None when nothing
    survived review.
    """
    accepted = set(consensus.accepted_claim_ids)
    items: list[MigrationKnowledgeItem] = []
    used_source_ids: set[str] = set()
    for claim in sorted(research.claims, key=lambda item: item.id):
        if claim.id not in accepted:
            continue
        if not claim.field_path.startswith("migration_behavior"):
            continue
        if not isinstance(claim.value, dict):
            continue
        item = MigrationKnowledgeItem.model_validate(claim.value)
        items.append(item)
        used_source_ids.update(claim.sources)
    if not items:
        return None
    sources = [source for source in research.sources if source.id in used_source_ids]
    return MigrationKnowledge(
        id=knowledge_id,
        source_model=source_model,
        target_model=target_model,
        changes=items,
        sources=sources,
        checked_at=research.research_date,
    )
