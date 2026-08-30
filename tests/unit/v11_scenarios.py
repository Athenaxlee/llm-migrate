"""Deterministic fake-agent scenarios for V1.1 workflow tests.

The fake host never touches the network; it returns fixture artifacts so the
deterministic orchestration, review, consensus, and overlay logic can be
exercised end to end without any real agent runtime.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from llm_migrate.core.agent_research import (
    ClaimScope,
    ClaimVerdict,
    EvidenceReview,
    MigrationResearchRequest,
    ModelEndpointIdentity,
    ResearchScope,
    ResearchTopic,
    VerdictKind,
    artifact_sha256,
)
from llm_migrate.core.knowledge import EvidenceClaim, ResearchResult
from llm_migrate.core.models import SourceReference
from llm_migrate.core.orchestration import (
    AgentUsage,
    ResearchAssignment,
    ReviewAssignment,
)

AS_OF = date(2026, 8, 29)
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)

SOURCE_IDENTITY = ModelEndpointIdentity(
    provider="anthropic",
    platform="anthropic-api",
    model="claude-sonnet-5",
    endpoint="messages",
)
TARGET_IDENTITY = ModelEndpointIdentity(
    provider="openai",
    platform="openai-api",
    model="gpt-6.0-nova",
    endpoint="responses",
)


def _official_source(source_id: str, url_slug: str) -> SourceReference:
    return SourceReference.model_validate(
        {
            "id": source_id,
            "url": f"https://developers.example.com/docs/{url_slug}",
            "title": url_slug,
            "source_type": "official_documentation",
            "authority_tier": 2,
            "publisher": "Example Provider",
            "retrieved_at": AS_OF.isoformat(),
            "supports": ["identity", "platforms", "lifecycle", "pricing", "capabilities"],
        }
    )


def target_research() -> ResearchResult:
    """Official-evidence research for a model missing from the registry."""
    source = _official_source("official-nova-model-page", "gpt-6.0-nova")
    claims = [
        EvidenceClaim(
            id="nova-identity",
            field_path="identity",
            value={
                "canonical_name": "gpt-6.0-nova",
                "display_name": "GPT-6.0 Nova",
                "provider": "openai",
                "model_family": "gpt-6.0",
                "version": "nova",
                "aliases": ["gpt 6.0 nova"],
            },
            statement="The official model page identifies GPT-6.0 Nova.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-platform",
            field_path="platforms",
            value=[
                {
                    "platform": "openai-api",
                    "endpoint": "responses",
                    "model_id": "gpt-6.0-nova",
                }
            ],
            statement="Nova is available through the Responses API.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-lifecycle",
            field_path="lifecycle.status",
            value="active",
            statement="The catalog lists Nova as a current model.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-text-input",
            field_path="capabilities.text_input",
            value=True,
            statement="The official model page lists text input.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-tool-use",
            field_path="capabilities.tool_use",
            value=True,
            statement="The official model page lists tool use.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-structured-output",
            field_path="capabilities.structured_output",
            value=True,
            statement="The official model page lists structured outputs.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-streaming",
            field_path="capabilities.streaming",
            value=True,
            statement="The official model page lists streaming.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-context",
            field_path="capabilities.context_window_tokens",
            value=2_000_000,
            statement="The official model page lists a 2,000,000-token context window.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-output-limit",
            field_path="capabilities.maximum_output_tokens",
            value=256_000,
            statement="The official model page lists a 256,000-token output maximum.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-price-input",
            field_path="pricing.input",
            value={"amount": "6", "currency": "USD", "unit": "per_1m_tokens"},
            statement="The official model page lists six dollars per million input tokens.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="nova-price-output",
            field_path="pricing.output",
            value={"amount": "24", "currency": "USD", "unit": "per_1m_tokens"},
            statement="The official model page lists 24 dollars per million output tokens.",
            confidence="authoritative",
            sources=[source.id],
        ),
    ]
    return ResearchResult(
        subject="gpt-6.0-nova",
        canonical_model_candidate="gpt-6.0-nova",
        research_date=AS_OF,
        claims=claims,
        sources=[source],
    )


def source_research() -> ResearchResult:
    """No-change evidence refresh for the stale canonical source profile."""
    source = _official_source("official-sonnet-5-model-page", "claude-sonnet-5")
    claims = [
        EvidenceClaim(
            id="sonnet-5-lifecycle",
            field_path="lifecycle.status",
            value="active",
            statement="The official catalog still recommends Sonnet 5.",
            confidence="authoritative",
            sources=[source.id],
        ),
        EvidenceClaim(
            id="sonnet-5-price-input",
            field_path="pricing.input.amount",
            value="2",
            statement="The official pricing page still lists two dollars per million.",
            confidence="authoritative",
            sources=[source.id],
        ),
    ]
    return ResearchResult(
        subject="claude-sonnet-5",
        canonical_model_candidate="claude-sonnet-5",
        research_date=AS_OF,
        claims=claims,
        sources=[source],
    )


def pair_research() -> ResearchResult:
    """Pair-specific migration behavior; profiles alone never establish this."""
    source = _official_source("official-nova-migration-guide", "migrating-to-nova")
    claims = [
        EvidenceClaim(
            id="pair-tool-schema",
            field_path="migration_behavior.tools",
            value={
                "field_path": "tools",
                "statement": (
                    "Nova requires flattened tool schemas; nested anyOf branches "
                    "must be expanded before migration."
                ),
                "severity": "high",
                "recommended_action": "Flatten nested tool schemas during migration.",
                "supporting_sources": [source.id],
            },
            statement="The official migration guide documents a tool-schema change.",
            confidence="authoritative",
            sources=[source.id],
        ),
    ]
    return ResearchResult(
        subject="claude-sonnet-5 -> gpt-6.0-nova",
        canonical_model_candidate=None,
        research_date=AS_OF,
        claims=claims,
        sources=[source],
    )


def supportive_review(
    research: ResearchResult,
    *,
    reviewer_id: str,
    researcher_id: str,
    provider: str,
) -> EvidenceReview:
    """A reviewer that refetched every cited source and found each claim supported."""
    return EvidenceReview(
        reviewer_id=reviewer_id,
        researcher_id=researcher_id,
        research_sha256=artifact_sha256(research),
        reviewed_at=AS_OF,
        verdicts=[
            ClaimVerdict(
                claim_id=claim.id,
                verdict=VerdictKind.SUPPORTED,
                checked_source_ids=list(claim.sources),
                scope=ClaimScope(provider=provider),
                rationale="Refetched the cited official page; it states this fact.",
            )
            for claim in research.claims
        ],
    )


class FakeAgentHost:
    """User-supplied agent runtime double: fixture artifacts, recorded usage."""

    def __init__(
        self,
        research_by_scope: dict[ResearchScope, ResearchResult],
        *,
        usage_per_call: AgentUsage | None = None,
        review_overrides: dict[str, EvidenceReview] | None = None,
        research_failures: int = 0,
    ) -> None:
        self.research_by_scope = research_by_scope
        self.usage_per_call = usage_per_call or AgentUsage(input_tokens=900, output_tokens=100)
        self.review_overrides = review_overrides or {}
        self.research_failures = research_failures
        self.research_calls = 0
        self.review_calls = 0

    def run_research(
        self, assignment: ResearchAssignment
    ) -> tuple[str, ResearchResult, AgentUsage]:
        self.research_calls += 1
        if self.research_failures > 0:
            self.research_failures -= 1
            raise ValueError("fixture research agent failed")
        research = self.research_by_scope[assignment.scope]
        return f"researcher-{assignment.scope.value}", research, self.usage_per_call

    def run_review(self, assignment: ReviewAssignment) -> tuple[str, EvidenceReview, AgentUsage]:
        self.review_calls += 1
        override = self.review_overrides.get(assignment.subject)
        if override is not None:
            return "reviewer-fixed", override, self.usage_per_call
        provider = (
            SOURCE_IDENTITY.provider
            if assignment.subject == SOURCE_IDENTITY.model
            else TARGET_IDENTITY.provider
        )
        review = supportive_review(
            assignment.research,
            reviewer_id="reviewer-independent",
            researcher_id=assignment.researcher_id,
            provider=provider,
        )
        return "reviewer-independent", review, self.usage_per_call


def standard_host() -> FakeAgentHost:
    return FakeAgentHost(
        {
            ResearchScope.SOURCE: source_research(),
            ResearchScope.TARGET: target_research(),
            ResearchScope.PAIR: pair_research(),
        }
    )


def standard_request(
    requirements_sha256: str, run_id: str = "run-nova-001"
) -> MigrationResearchRequest:
    return MigrationResearchRequest(
        run_id=run_id,
        source=SOURCE_IDENTITY,
        target=TARGET_IDENTITY,
        application_requirements_sha256=requirements_sha256,
        topics=[
            ResearchTopic.PRICING,
            ResearchTopic.LIFECYCLE,
            ResearchTopic.AVAILABILITY,
            ResearchTopic.CAPABILITIES,
            ResearchTopic.MIGRATION_BEHAVIOR,
        ],
        as_of=AS_OF,
    )
