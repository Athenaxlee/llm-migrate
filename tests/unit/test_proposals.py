from __future__ import annotations

from datetime import date
from pathlib import Path

from llm_migrate.core.artifacts import write_proposal_artifacts
from llm_migrate.core.knowledge import (
    ChangeClassification,
    Confidence,
    EmpiricalObservation,
    EvidenceClaim,
    EvidenceConflict,
    EvidenceKind,
    ResearchResult,
)
from llm_migrate.core.models import SourceReference
from llm_migrate.service import MigrationService


def _source(source_id: str, tier: int, source_type: str) -> SourceReference:
    return SourceReference.model_validate(
        {
            "id": source_id,
            "url": f"https://example.com/{source_id}",
            "title": source_id,
            "source_type": source_type,
            "authority_tier": tier,
            "publisher": "Example",
            "retrieved_at": "2026-08-08",
            "supports": ["capabilities", "parameters"],
        }
    )


def research_result() -> ResearchResult:
    claims = [
        EvidenceClaim(
            id="unchanged",
            field_path="capabilities.tool_use",
            value=True,
            statement="Tool use remains supported.",
            confidence=Confidence.HIGH,
            sources=["official"],
        ),
        EvidenceClaim(
            id="updated",
            field_path="capabilities.context_window_tokens",
            value=250000,
            statement="The context limit changed.",
            confidence=Confidence.AUTHORITATIVE,
            sources=["official"],
        ),
        EvidenceClaim(
            id="added",
            field_path="parameters.new_control",
            value={"state": "supported"},
            statement="A new control exists.",
            confidence=Confidence.HIGH,
            sources=["official"],
        ),
        EvidenceClaim(
            id="conflict-yes",
            field_path="capabilities.reasoning",
            value=True,
            statement="Reasoning is supported.",
            confidence=Confidence.HIGH,
            sources=["official"],
        ),
        EvidenceClaim(
            id="conflict-no",
            field_path="capabilities.reasoning",
            value=False,
            statement="Reasoning is unavailable on one endpoint.",
            confidence=Confidence.HIGH,
            sources=["official"],
        ),
        EvidenceClaim(
            id="weak",
            field_path="capabilities.telepathy",
            value=True,
            statement="A community user reports telepathy.",
            kind=EvidenceKind.OBSERVATION,
            confidence=Confidence.LOW,
            sources=["community"],
        ),
    ]
    return ResearchResult(
        subject="Fixture Alpha evidence update",
        canonical_model_candidate="alpha large",
        research_date=date(2026, 8, 8),
        claims=claims,
        sources=[
            _source("official", 2, "official_documentation"),
            _source("community", 6, "community_report"),
        ],
        observations=[
            EmpiricalObservation(
                id="eager-tools",
                topic="tool_eagerness",
                statement="A user observed eager tool calls.",
                sources=["community"],
            )
        ],
        conflicts=[
            EvidenceConflict(
                id="reasoning-conflict",
                claim_ids=["conflict-yes", "conflict-no"],
                statement="Endpoint evidence conflicts.",
            )
        ],
    )


def test_proposal_classifies_claims_and_preserves_canonical_registry(
    service: MigrationService,
) -> None:
    before = service.resolve_model("alpha large")
    proposal = service.propose_registry_update(research_result())
    after = service.resolve_model("alpha large")

    assert before == after
    assert before.capabilities.context_window_tokens == 200000
    assert proposal.changed_facts[0].classification is ChangeClassification.UPDATE
    assert proposal.new_facts[0].classification is ChangeClassification.ADD
    assert proposal.unchanged_facts[0].classification is ChangeClassification.NO_CHANGE
    assert len(proposal.conflicting_facts) == 2
    assert (
        proposal.unsupported_claims[0].classification is ChangeClassification.INSUFFICIENT_EVIDENCE
    )
    assert proposal.new_observations[0].topic == "tool_eagerness"
    assert proposal.candidate_model is not None
    assert proposal.candidate_model["capabilities"]["context_window_tokens"] == 250000
    candidate_source_ids = {item["id"] for item in proposal.candidate_model["sources"]}
    assert "official" in candidate_source_ids
    assert "community" not in candidate_source_ids


def test_candidate_artifacts_are_written_only_when_requested(
    service: MigrationService, tmp_path: Path
) -> None:
    research = research_result()
    proposal = service.propose_registry_update(research)
    assert list(tmp_path.iterdir()) == []
    target = write_proposal_artifacts(proposal, research, tmp_path)
    assert {path.name for path in target.iterdir()} == {
        "proposal.yaml",
        "proposal.md",
        "evidence.yaml",
        "candidate-model.yaml",
    }
    proposal_yaml = (target / "proposal.yaml").read_text(encoding="utf-8")
    assert "model: alpha large" in proposal_yaml
    assert "changes:" in proposal_yaml
    assert "conflicts:" in proposal_yaml
    assert "unknown_fields:" in proposal_yaml


def test_proposal_artifacts_match_exact_goldens(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    research = research_result()
    proposal = service.propose_registry_update(research)
    target = write_proposal_artifacts(proposal, research, tmp_path)
    golden = project_root / "tests" / "golden" / "proposals" / target.name
    generated_names = sorted(path.name for path in target.iterdir())
    assert generated_names == sorted(path.name for path in golden.iterdir())
    for name in generated_names:
        assert (target / name).read_text(encoding="utf-8") == (golden / name).read_text(
            encoding="utf-8"
        ), f"{name} no longer matches its golden snapshot"
