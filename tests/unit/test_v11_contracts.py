from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_migrate.core.agent_research import (
    ClaimScope,
    ClaimVerdict,
    ConsensusError,
    DisputeArbitration,
    EvidenceReview,
    MigrationResearchRequest,
    ResearchTopic,
    SessionSuitability,
    TopicCoverage,
    TrustLevel,
    VerdictKind,
    arbitrate_conflict,
    artifact_sha256,
    build_research_consensus,
    compile_migration_knowledge_candidate,
)
from llm_migrate.core.knowledge import EvidenceConflict, ResearchResult
from llm_migrate.core.orchestration import OrchestrationRun, StoppingReason
from llm_migrate.core.session import SessionRegistryManifest
from llm_migrate.service import MigrationService
from tests.unit.v11_scenarios import (
    AS_OF,
    NOW,
    SOURCE_IDENTITY,
    TARGET_IDENTITY,
    pair_research,
    standard_request,
    supportive_review,
    target_research,
)

REQUIREMENTS_SHA = "0" * 64


def _review(research: ResearchResult) -> EvidenceReview:
    return supportive_review(
        research,
        reviewer_id="reviewer-independent",
        researcher_id="researcher-target",
        provider="openai",
    )


def test_new_contracts_reject_unknown_fields() -> None:
    request = standard_request(REQUIREMENTS_SHA)
    payload = request.model_dump(mode="json")
    payload["surprise"] = True
    with pytest.raises(ValidationError):
        MigrationResearchRequest.model_validate(payload)

    review = _review(target_research())
    review_payload = review.model_dump(mode="json")
    review_payload["auto_approve"] = True
    with pytest.raises(ValidationError):
        EvidenceReview.model_validate(review_payload)

    with pytest.raises(ValidationError):
        OrchestrationRun.model_validate(
            {
                "run_id": "run-x",
                "request_sha256": REQUIREMENTS_SHA,
                "stopping_reason": "completed",
                "hidden_reasoning": "must not exist",
            }
        )

    manifest_payload = {
        "run_id": "run-x",
        "created_at": NOW.isoformat(),
        "expires_at": NOW.isoformat(),
        "base_registry_sha256": REQUIREMENTS_SHA,
        "request_sha256": REQUIREMENTS_SHA,
        "trust_level": "session_agent_reviewed",
        "credential": "nope",
    }
    with pytest.raises(ValidationError):
        SessionRegistryManifest.model_validate(manifest_payload)


def test_artifact_hashes_are_stable_and_round_trip() -> None:
    research = target_research()
    first = artifact_sha256(research)
    round_tripped = ResearchResult.model_validate(research.model_dump(mode="json"))
    assert artifact_sha256(round_tripped) == first


def test_reviewer_cannot_review_own_result() -> None:
    research = target_research()
    with pytest.raises(ValidationError, match="cannot review its own"):
        supportive_review(
            research,
            reviewer_id="agent-a",
            researcher_id="agent-a",
            provider="openai",
        )


def test_supported_verdict_requires_checked_sources() -> None:
    with pytest.raises(ValidationError, match="independently checked sources"):
        ClaimVerdict(
            claim_id="x",
            verdict=VerdictKind.SUPPORTED,
            checked_source_ids=[],
            scope=ClaimScope(provider="openai"),
            rationale="looks fine",
        )


def test_consensus_is_deterministic_and_ordered() -> None:
    research = target_research()
    review = _review(research)
    request = standard_request(REQUIREMENTS_SHA)
    first = build_research_consensus(research, review, request.topics)
    second = build_research_consensus(research, review, request.topics)
    assert first == second
    assert first.accepted_claim_ids == sorted(first.accepted_claim_ids)
    assert first.suitability is SessionSuitability.SUITABLE
    assert first.trust_level is TrustLevel.SESSION_AGENT_REVIEWED
    assert first.topic_coverage[ResearchTopic.PRICING] is TopicCoverage.COVERED
    assert first.topic_coverage[ResearchTopic.MIGRATION_BEHAVIOR] is TopicCoverage.MISSING


def test_consensus_rejects_mismatched_review_hash() -> None:
    research = target_research()
    review = _review(research).model_copy(update={"research_sha256": "0" * 64})
    with pytest.raises(ConsensusError, match="does not reference"):
        build_research_consensus(research, review, [ResearchTopic.PRICING])


def test_missing_verdict_fails_closed() -> None:
    research = target_research()
    review = _review(research)
    partial = review.model_copy(update={"verdicts": review.verdicts[:-1]})
    consensus = build_research_consensus(research, partial, [ResearchTopic.PRICING])
    assert consensus.suitability is SessionSuitability.UNSUITABLE
    assert consensus.trust_level is TrustLevel.SESSION_UNREVIEWED
    assert any("no independent review verdict" in blocker for blocker in consensus.blockers)


def test_reviewer_must_check_a_cited_source() -> None:
    research = target_research()
    review = _review(research)
    lazy = review.model_copy(
        update={
            "verdicts": [
                verdict.model_copy(update={"checked_source_ids": ["some-other-page"]})
                for verdict in review.verdicts
            ]
        }
    )
    consensus = build_research_consensus(research, lazy, [ResearchTopic.PRICING])
    assert consensus.accepted_claim_ids == []
    assert all("did not independently check" in item.reason for item in consensus.rejected_claims)


def test_agent_agreement_is_not_evidence_for_conflicts() -> None:
    """Two contradictory claims stay unresolved even when both are 'supported'."""
    base = target_research()
    duplicate = base.claims[-1].model_copy(
        update={
            "id": "nova-price-output-alt",
            "value": {"amount": "30", "currency": "USD", "unit": "per_1m_tokens"},
        }
    )
    conflicted = base.model_copy(
        update={
            "claims": [*base.claims, duplicate],
            "conflicts": [
                EvidenceConflict(
                    id="output-price-conflict",
                    claim_ids=["nova-price-output", "nova-price-output-alt"],
                    statement="Two official pages disagree on the output price.",
                )
            ],
        }
    )
    review = _review(conflicted)
    consensus = build_research_consensus(conflicted, review, [ResearchTopic.PRICING])
    assert "nova-price-output" not in consensus.accepted_claim_ids
    assert "nova-price-output-alt" not in consensus.accepted_claim_ids
    assert consensus.unresolved_conflicts
    assert consensus.topic_coverage[ResearchTopic.PRICING] is TopicCoverage.BLOCKED
    assert consensus.suitability is SessionSuitability.UNSUITABLE


def test_arbiter_requires_independent_qualifying_evidence() -> None:
    base = target_research()
    duplicate = base.claims[-1].model_copy(
        update={
            "id": "nova-price-output-alt",
            "value": {"amount": "30", "currency": "USD", "unit": "per_1m_tokens"},
        }
    )
    conflicted = base.model_copy(
        update={
            "claims": [*base.claims, duplicate],
            "conflicts": [
                EvidenceConflict(
                    id="output-price-conflict",
                    claim_ids=["nova-price-output", "nova-price-output-alt"],
                    statement="Two official pages disagree on the output price.",
                )
            ],
        }
    )
    review = _review(conflicted)
    consensus = build_research_consensus(conflicted, review, [ResearchTopic.PRICING])

    bare_arbitration = DisputeArbitration(
        arbiter_id="arbiter-1",
        conflict_claim_ids=["nova-price-output", "nova-price-output-alt"],
        upheld_claim_id="nova-price-output",
        rationale="I simply prefer this one.",
    )
    with pytest.raises(ConsensusError, match="independently sourced evidence"):
        arbitrate_conflict(consensus, conflicted, review, bare_arbitration)

    with pytest.raises(ConsensusError, match="independent of researcher and reviewer"):
        arbitrate_conflict(
            consensus,
            conflicted,
            review,
            bare_arbitration.model_copy(update={"arbiter_id": "reviewer-independent"}),
        )

    evidence = target_research()
    corroborating = evidence.model_copy(
        update={
            "subject": "gpt-6.0-nova output price arbitration",
            "claims": [claim for claim in evidence.claims if claim.id == "nova-price-output"],
        }
    )
    resolved = arbitrate_conflict(
        consensus,
        conflicted,
        review,
        DisputeArbitration(
            arbiter_id="arbiter-1",
            conflict_claim_ids=["nova-price-output", "nova-price-output-alt"],
            upheld_claim_id="nova-price-output",
            added_evidence=corroborating,
            rationale="A separately retrieved official page confirms 24 dollars.",
        ),
    )
    assert "nova-price-output" in resolved.accepted_claim_ids
    assert not resolved.unresolved_conflicts
    assert resolved.suitability is SessionSuitability.SUITABLE


def test_pair_candidate_requires_accepted_pair_claims(service: MigrationService) -> None:
    research = pair_research()
    review = supportive_review(
        research,
        reviewer_id="reviewer-independent",
        researcher_id="researcher-pair",
        provider="openai",
    )
    consensus = build_research_consensus(research, review, [ResearchTopic.MIGRATION_BEHAVIOR])
    candidate = compile_migration_knowledge_candidate(
        "session-run-x-migration",
        SOURCE_IDENTITY.model,
        TARGET_IDENTITY.model,
        research,
        consensus,
    )
    assert candidate is not None
    assert candidate.changes[0].severity.value == "high"

    empty_consensus = consensus.model_copy(update={"accepted_claim_ids": []})
    assert (
        compile_migration_knowledge_candidate(
            "session-run-x-migration",
            SOURCE_IDENTITY.model,
            TARGET_IDENTITY.model,
            research,
            empty_consensus,
        )
        is None
    )


def test_service_stage_validators_flag_scope_and_integrity_problems(
    service: MigrationService,
) -> None:
    request = standard_request(REQUIREMENTS_SHA)
    research = target_research()
    assert service.validate_research_result(research, request) == []

    off_subject = research.model_copy(update={"subject": "some-unrelated-model"})
    problems = service.validate_research_result(off_subject, request)
    assert any("not one of the requested identities" in problem for problem in problems)

    late = research.model_copy(update={"research_date": AS_OF.replace(year=2027)})
    problems = service.validate_research_result(late, request)
    assert any("after the requested as-of date" in problem for problem in problems)

    review = _review(research)
    assert service.validate_evidence_review(review, research) == []
    partial = review.model_copy(update={"verdicts": review.verdicts[1:]})
    problems = service.validate_evidence_review(partial, research)
    assert any("has no review verdict" in problem for problem in problems)


def test_run_record_rejects_unknown_stopping_reason() -> None:
    with pytest.raises(ValidationError):
        OrchestrationRun.model_validate(
            {
                "run_id": "run-x",
                "request_sha256": REQUIREMENTS_SHA,
                "stopping_reason": "vibes",
            }
        )
    run = OrchestrationRun(
        run_id="run-x",
        request_sha256=REQUIREMENTS_SHA,
        stopping_reason=StoppingReason.COMPLETED,
    )
    assert run.total_input_tokens == 0
