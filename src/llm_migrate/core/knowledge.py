"""Evidence, research, migration-knowledge, and registry proposal schemas."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, HttpUrl, model_validator

from llm_migrate.core.models import (
    ComparisonSeverity,
    FreshnessCategory,
    SourceReference,
    StrictModel,
)


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTHORITATIVE = "authoritative"


class EvidenceKind(StrEnum):
    FACT = "fact"
    OBSERVATION = "observation"
    INFERENCE = "inference"


class EvidenceType(StrEnum):
    DIRECT = "direct"
    DERIVED = "derived"
    EMPIRICAL = "empirical"


class ClaimAction(StrEnum):
    SET = "set"
    REMOVE = "remove"


class EvidenceClaim(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    field_path: str = Field(min_length=1)
    value: Any | None = None
    statement: str = Field(min_length=1)
    kind: EvidenceKind = EvidenceKind.FACT
    evidence_type: EvidenceType = EvidenceType.DIRECT
    confidence: Confidence
    sources: list[str] = Field(default_factory=list)
    action: ClaimAction = ClaimAction.SET
    notes: str | None = None


class EmpiricalObservation(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    topic: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    confidence: Confidence = Confidence.LOW
    sources: list[str] = Field(default_factory=list)
    model: str | None = None
    notes: str | None = None


class EvidenceConflict(StrictModel):
    id: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=2)
    statement: str = Field(min_length=1)
    notes: str | None = None


class ResearchResult(StrictModel):
    schema_version: Literal["1"] = "1"
    subject: str = Field(min_length=1)
    canonical_model_candidate: str | None = None
    research_date: date
    claims: list[EvidenceClaim] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    observations: list[EmpiricalObservation] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> ResearchResult:
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("research source IDs must be unique")
        claim_ids = [claim.id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("research claim IDs must be unique")
        known_sources = set(source_ids)
        for claim in self.claims:
            missing = set(claim.sources) - known_sources
            if missing:
                raise ValueError(f"unknown source references: {sorted(missing)}")
        for observation in self.observations:
            missing = set(observation.sources) - known_sources
            if missing:
                raise ValueError(f"unknown source references: {sorted(missing)}")
        known_claims = set(claim_ids)
        for conflict in self.conflicts:
            missing = set(conflict.claim_ids) - known_claims
            if missing:
                raise ValueError(f"unknown conflict claim references: {sorted(missing)}")
        return self


class MigrationKnowledgeItem(StrictModel):
    field_path: str
    statement: str
    severity: ComparisonSeverity
    recommended_action: str | None = None
    supporting_sources: list[str] = Field(min_length=1)


class MigrationKnowledge(StrictModel):
    schema_version: Literal["1"] = "1"
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source_model: str
    target_model: str
    changes: list[MigrationKnowledgeItem] = Field(min_length=1)
    sources: list[SourceReference] = Field(min_length=1)
    checked_at: date

    @model_validator(mode="after")
    def validate_source_references(self) -> MigrationKnowledge:
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("migration source IDs must be unique")
        known = set(source_ids)
        for change in self.changes:
            missing = set(change.supporting_sources) - known
            if missing:
                raise ValueError(f"unknown migration source references: {sorted(missing)}")
        return self


class ObservationRecord(StrictModel):
    schema_version: Literal["1"] = "1"
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model: str | None = None
    topic: str
    statement: str
    confidence: Confidence
    sources: list[SourceReference] = Field(min_length=1)
    notes: str | None = None


class ChangeClassification(StrEnum):
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"
    NO_CHANGE = "no_change"
    CONFLICT = "conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ChangeRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProposedFactChange(StrictModel):
    field_path: str
    current_value: Any | None = None
    proposed_value: Any | None = None
    classification: ChangeClassification
    confidence: Confidence
    supporting_sources: list[str]
    reason: str
    risk: ChangeRisk


class FreshnessUpdate(StrictModel):
    category: FreshnessCategory
    previous_checked_at: date | None = None
    proposed_checked_at: date


class RegistryUpdateProposal(StrictModel):
    schema_version: Literal["1"] = "1"
    subject: str
    canonical_model_candidate: str | None = None
    proposal_date: date
    new_facts: list[ProposedFactChange] = Field(default_factory=list)
    changed_facts: list[ProposedFactChange] = Field(default_factory=list)
    unchanged_facts: list[ProposedFactChange] = Field(default_factory=list)
    conflicting_facts: list[ProposedFactChange] = Field(default_factory=list)
    unsupported_claims: list[ProposedFactChange] = Field(default_factory=list)
    new_sources: list[SourceReference] = Field(default_factory=list)
    new_observations: list[EmpiricalObservation] = Field(default_factory=list)
    freshness_updates: list[FreshnessUpdate] = Field(default_factory=list)
    candidate_model: dict[str, Any] | None = None
    candidate_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StaleRegistrySection(StrictModel):
    canonical_name: str
    category: FreshnessCategory
    checked_at: date | None = None
    max_age_days: int | None = None
    stale: bool


class RegistryFreshnessReport(StrictModel):
    as_of_date: date
    sections: list[StaleRegistrySection]


class RegistryValidationReport(StrictModel):
    valid: bool
    model_count: int
    migration_count: int
    observation_count: int
    warnings: list[str] = Field(default_factory=list)


class ResearchCacheEntry(StrictModel):
    """Metadata-only placeholder for a future opt-in local research cache."""

    url: HttpUrl
    retrieved_at: date
    etag: str | None = None
    last_modified: str | None = None
    content_hash: str
    extracted_claim_ids: list[str] = Field(default_factory=list)
