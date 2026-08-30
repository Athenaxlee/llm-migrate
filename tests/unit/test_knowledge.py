from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from llm_migrate.core.knowledge import (
    Confidence,
    EmpiricalObservation,
    EvidenceClaim,
    EvidenceConflict,
    EvidenceKind,
    ResearchResult,
)
from llm_migrate.core.models import SourceReference


def source(source_id: str = "official-source", tier: int = 2) -> SourceReference:
    return SourceReference.model_validate(
        {
            "id": source_id,
            "url": "https://example.com/evidence",
            "title": "Evidence",
            "source_type": "official_documentation" if tier <= 4 else "community_report",
            "authority_tier": tier,
            "publisher": "Example publisher",
            "retrieved_at": "2026-08-08",
            "supports": ["capabilities.tool_use"],
        }
    )


def test_source_schema_rejects_invalid_authority_tier() -> None:
    with pytest.raises(ValidationError, match="authority_tier"):
        source(tier=7)


def test_fact_and_observation_are_explicit() -> None:
    fact = EvidenceClaim(
        id="fact",
        field_path="capabilities.tool_use",
        value=True,
        statement="Tool use is supported.",
        kind=EvidenceKind.FACT,
        confidence=Confidence.HIGH,
        sources=["official-source"],
    )
    observation = EmpiricalObservation(
        id="observation",
        topic="tool_eagerness",
        statement="A contributor observed eager tool use.",
        sources=["official-source"],
    )
    assert fact.kind is EvidenceKind.FACT
    assert observation.confidence is Confidence.LOW


def test_research_result_allows_contradictions_but_validates_references() -> None:
    claims = [
        EvidenceClaim(
            id=f"claim-{str(value).lower()}",
            field_path="capabilities.structured_output",
            value=value,
            statement=f"Structured output is {value}.",
            confidence=Confidence.HIGH,
            sources=["official-source"],
        )
        for value in (True, False)
    ]
    result = ResearchResult(
        subject="Contradictory platform evidence",
        research_date=date(2026, 8, 8),
        claims=claims,
        sources=[source()],
        conflicts=[
            EvidenceConflict(
                id="platform-conflict",
                claim_ids=["claim-true", "claim-false"],
                statement="Different platforms expose different behavior.",
            )
        ],
    )
    assert len(result.claims) == 2

    with pytest.raises(ValidationError, match="unknown source references"):
        ResearchResult(
            subject="Broken references",
            research_date=date(2026, 8, 8),
            claims=[claims[0].model_copy(update={"sources": ["missing"]})],
            sources=[source()],
        )
