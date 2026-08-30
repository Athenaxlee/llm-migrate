"""Pure evidence normalization and registry update proposal generation."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from llm_migrate.core.knowledge import (
    ChangeClassification,
    ChangeRisk,
    ClaimAction,
    Confidence,
    EvidenceClaim,
    EvidenceKind,
    FreshnessUpdate,
    ProposedFactChange,
    RegistryUpdateProposal,
    ResearchResult,
)
from llm_migrate.core.models import FreshnessCategory, FreshnessMetadata, ModelProfile
from llm_migrate.core.registry import ModelRegistry
from llm_migrate.core.resolver import ModelNotFoundError, resolve_model

_MISSING = object()
_DEFAULT_FRESHNESS_DAYS = {
    FreshnessCategory.PRICING: 7,
    FreshnessCategory.LIFECYCLE: 7,
    FreshnessCategory.AVAILABILITY: 14,
    FreshnessCategory.CAPABILITIES: 30,
    FreshnessCategory.PROMPTING_GUIDANCE: 30,
}
_FIXED_NESTED_FIELDS = {
    "identity": {
        "canonical_name",
        "display_name",
        "provider",
        "model_family",
        "version",
        "aliases",
    },
    "lifecycle": {
        "status",
        "announced_on",
        "deprecated_on",
        "end_of_life_on",
        "earliest_end_of_life_on",
        "sources",
    },
    "capabilities": {
        "text_input",
        "image_input",
        "document_input",
        "structured_output",
        "tool_use",
        "parallel_tool_use",
        "reasoning",
        "prompt_caching",
        "streaming",
        "batch_inference",
        "context_window_tokens",
        "maximum_output_tokens",
        "sources",
    },
    "prompt_guidance": {
        "preferred_structure",
        "delimiter_guidance",
        "reasoning_guidance",
        "verbosity_sensitivity",
        "negative_instruction_guidance",
        "structured_output_guidance",
        "tool_use_guidance",
        "sources",
    },
}


def _lookup(data: dict[str, Any], field_path: str) -> Any:
    current: Any = data
    for segment in field_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _set_value(data: dict[str, Any], field_path: str, value: Any) -> None:
    segments = field_path.split(".")
    current = data
    for segment in segments[:-1]:
        nested = current.setdefault(segment, {})
        if not isinstance(nested, dict):
            raise ValueError(f"cannot set nested field below non-mapping {segment!r}")
        current = nested
    current[segments[-1]] = deepcopy(value)


def _remove_value(data: dict[str, Any], field_path: str) -> None:
    segments = field_path.split(".")
    current: Any = data
    for segment in segments[:-1]:
        if not isinstance(current, dict) or segment not in current:
            return
        current = current[segment]
    if isinstance(current, dict):
        current.pop(segments[-1], None)


def _is_supported(claim: EvidenceClaim, source_tiers: dict[str, int]) -> tuple[bool, str]:
    if claim.kind is not EvidenceKind.FACT:
        return False, f"{claim.kind.value} cannot be promoted to a canonical model fact"
    if not claim.sources:
        return False, "claim has no supporting source"
    tiers = [source_tiers[source_id] for source_id in claim.sources]
    if claim.confidence is Confidence.AUTHORITATIVE and min(tiers) <= 3:
        return True, "authoritative claim supported by a tier 1-3 source"
    if claim.confidence is Confidence.HIGH and min(tiers) <= 2:
        return True, "high-confidence claim supported by a tier 1-2 source"
    return False, "evidence does not meet the canonical fact threshold"


def _risk(field_path: str, action: ClaimAction) -> ChangeRisk:
    high = (
        "pricing",
        "lifecycle",
        "platforms",
        "identity.canonical_name",
        "capabilities.context_window_tokens",
        "capabilities.maximum_output_tokens",
    )
    if field_path.startswith(high):
        return ChangeRisk.HIGH
    if action is ClaimAction.REMOVE and field_path.startswith(("parameters", "capabilities")):
        return ChangeRisk.HIGH
    if field_path.startswith(("parameters", "capabilities")):
        return ChangeRisk.MEDIUM
    return ChangeRisk.LOW


def _freshness_category(field_path: str) -> FreshnessCategory | None:
    if field_path.startswith("pricing"):
        return FreshnessCategory.PRICING
    if field_path.startswith("lifecycle"):
        return FreshnessCategory.LIFECYCLE
    if field_path.startswith("platforms"):
        return FreshnessCategory.AVAILABILITY
    if field_path.startswith("capabilities") or field_path.startswith("parameters"):
        return FreshnessCategory.CAPABILITIES
    if field_path.startswith("prompt_guidance"):
        return FreshnessCategory.PROMPTING_GUIDANCE
    return None


def _serialized(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _field_path_allowed(field_path: str) -> bool:
    segments = field_path.split(".")
    root = segments[0]
    if root not in ModelProfile.model_fields:
        return False
    if len(segments) == 1:
        return True
    if root in _FIXED_NESTED_FIELDS:
        return segments[1] in _FIXED_NESTED_FIELDS[root] and len(segments) == 2
    if root == "pricing":
        if segments[1] not in {"input", "output", "cached_input", "batch"}:
            return False
        return len(segments) == 2 or (
            len(segments) == 3
            and segments[2]
            in {"amount", "currency", "unit", "valid_from", "valid_until", "notes", "sources"}
        )
    if root == "parameters":
        return len(segments) == 2 or (
            len(segments) == 3 and segments[2] in {"state", "default", "notes", "sources"}
        )
    if root == "freshness":
        return len(segments) in {2, 3} and segments[1] in {
            category.value for category in FreshnessCategory
        }
    return False


def _change(
    claim: EvidenceClaim,
    current: Any,
    classification: ChangeClassification,
    reason: str,
) -> ProposedFactChange:
    return ProposedFactChange(
        field_path=claim.field_path,
        current_value=None if current is _MISSING else current,
        proposed_value=None if claim.action is ClaimAction.REMOVE else claim.value,
        classification=classification,
        confidence=claim.confidence,
        supporting_sources=sorted(claim.sources),
        reason=reason,
        risk=_risk(claim.field_path, claim.action),
    )


def _source_ids(data: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(data, dict):
        if "source_type" in data and isinstance(data.get("id"), str):
            found.add(data["id"])
        for value in data.values():
            found.update(_source_ids(value))
    elif isinstance(data, list):
        for value in data:
            found.update(_source_ids(value))
    return found


def propose_registry_update(
    research: ResearchResult,
    registry: ModelRegistry,
) -> RegistryUpdateProposal:
    """Create a review artifact without mutating the canonical registry."""
    current_profile: ModelProfile | None = None
    if research.canonical_model_candidate:
        try:
            current_profile = resolve_model(registry, research.canonical_model_candidate).profile
        except ModelNotFoundError:
            current_profile = None
    current_data = current_profile.model_dump(mode="json") if current_profile else {}
    candidate: dict[str, Any] = deepcopy(current_data)
    if not candidate:
        candidate = {"schema_version": "1", "fixture": False, "sources": []}

    source_tiers = {source.id: int(source.authority_tier.value) for source in research.sources}
    explicit_conflicts = {
        claim_id for conflict in research.conflicts for claim_id in conflict.claim_ids
    }
    grouped: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in research.claims:
        grouped[claim.field_path].append(claim)

    groups: dict[ChangeClassification, list[ProposedFactChange]] = defaultdict(list)
    accepted_categories: set[FreshnessCategory] = set()
    accepted_source_ids: set[str] = set()
    for field_path in sorted(grouped):
        claims = sorted(grouped[field_path], key=lambda item: item.id)
        current = _lookup(current_data, field_path)
        contradictory = len(
            {(_serialized(claim.value), claim.action.value) for claim in claims}
        ) > 1 or any(claim.id in explicit_conflicts for claim in claims)
        if contradictory:
            for claim in claims:
                groups[ChangeClassification.CONFLICT].append(
                    _change(
                        claim,
                        current,
                        ChangeClassification.CONFLICT,
                        "contradictory claims require human resolution",
                    )
                )
            continue
        claim = claims[0]
        if not _field_path_allowed(field_path):
            groups[ChangeClassification.INSUFFICIENT_EVIDENCE].append(
                _change(
                    claim,
                    current,
                    ChangeClassification.INSUFFICIENT_EVIDENCE,
                    "field path is not part of the canonical ModelProfile schema",
                )
            )
            continue
        supported, reason = _is_supported(claim, source_tiers)
        if not supported:
            groups[ChangeClassification.INSUFFICIENT_EVIDENCE].append(
                _change(
                    claim,
                    current,
                    ChangeClassification.INSUFFICIENT_EVIDENCE,
                    reason,
                )
            )
            continue
        accepted_source_ids.update(claim.sources)
        if claim.action is ClaimAction.REMOVE:
            classification = (
                ChangeClassification.NO_CHANGE
                if current is _MISSING
                else ChangeClassification.REMOVE
            )
        elif current is _MISSING:
            classification = ChangeClassification.ADD
        elif current == claim.value:
            classification = ChangeClassification.NO_CHANGE
        else:
            classification = ChangeClassification.UPDATE
        groups[classification].append(_change(claim, current, classification, reason))
        if classification is not ChangeClassification.NO_CHANGE:
            if claim.action is ClaimAction.REMOVE:
                _remove_value(candidate, field_path)
            else:
                _set_value(candidate, field_path, claim.value)
        category = _freshness_category(field_path)
        if category:
            accepted_categories.add(category)

    existing_sources = _source_ids(current_data)
    new_sources = sorted(
        (source for source in research.sources if source.id not in existing_sources),
        key=lambda source: source.id,
    )
    merged_sources = list(candidate.get("sources", []))
    merged_ids = {item["id"] for item in merged_sources if isinstance(item, dict)}
    for source in new_sources:
        if source.id in accepted_source_ids and source.id not in merged_ids:
            merged_sources.append(source.model_dump(mode="json"))
    candidate["sources"] = merged_sources

    freshness_updates: list[FreshnessUpdate] = []
    existing_freshness = current_profile.freshness if current_profile else {}
    candidate_freshness = dict(candidate.get("freshness", {}))
    for category in sorted(accepted_categories, key=str):
        previous = existing_freshness.get(category)
        metadata = FreshnessMetadata(
            checked_at=research.research_date,
            max_age_days=(previous.max_age_days if previous else _DEFAULT_FRESHNESS_DAYS[category]),
        )
        candidate_freshness[category.value] = metadata.model_dump(mode="json")
        freshness_updates.append(
            FreshnessUpdate(
                category=category,
                previous_checked_at=previous.checked_at if previous else None,
                proposed_checked_at=research.research_date,
            )
        )
    candidate["freshness"] = candidate_freshness

    candidate_errors: list[str] = []
    candidate_model: dict[str, Any] | None
    try:
        candidate_model = ModelProfile.model_validate(candidate).model_dump(mode="json")
    except ValidationError as exc:
        candidate_model = None
        candidate_errors = [error["msg"] for error in exc.errors()]

    return RegistryUpdateProposal(
        subject=research.subject,
        canonical_model_candidate=research.canonical_model_candidate,
        proposal_date=research.research_date,
        new_facts=groups[ChangeClassification.ADD],
        changed_facts=[
            *groups[ChangeClassification.UPDATE],
            *groups[ChangeClassification.REMOVE],
        ],
        unchanged_facts=groups[ChangeClassification.NO_CHANGE],
        conflicting_facts=groups[ChangeClassification.CONFLICT],
        unsupported_claims=groups[ChangeClassification.INSUFFICIENT_EVIDENCE],
        new_sources=new_sources,
        new_observations=sorted(research.observations, key=lambda item: item.id),
        freshness_updates=freshness_updates,
        candidate_model=candidate_model,
        candidate_errors=candidate_errors,
        warnings=[*research.warnings, *research.unresolved_questions],
    )
