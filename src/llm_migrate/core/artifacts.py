"""Pure proposal renderers and explicitly requested artifact writing."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from llm_migrate.core.knowledge import (
    ProposedFactChange,
    RegistryUpdateProposal,
    ResearchResult,
)


def proposal_as_json(proposal: RegistryUpdateProposal) -> str:
    return proposal.model_dump_json(indent=2)


def proposal_as_yaml(proposal: RegistryUpdateProposal) -> str:
    return yaml.safe_dump(_proposal_document(proposal), sort_keys=False)


def _proposal_document(proposal: RegistryUpdateProposal) -> dict[str, object]:
    """Return the stable, machine-readable review contract.

    RegistryUpdateProposal keeps richer internal classifications for callers. The
    on-disk proposal deliberately exposes one ordered change stream so reviewers,
    agents, and CI do not need to understand those internal buckets.
    """
    changes = [*proposal.new_facts, *proposal.changed_facts]

    def fact(change: ProposedFactChange) -> dict[str, object]:
        payload = change.model_dump(mode="json")
        return {
            "field_path": payload["field_path"],
            "change_type": payload["classification"],
            "current_value": payload["current_value"],
            "proposed_value": payload["proposed_value"],
            "confidence": payload["confidence"],
            "risk": payload["risk"],
            "supporting_sources": payload["supporting_sources"],
            "reason": payload["reason"],
        }

    return {
        "schema_version": proposal.schema_version,
        "model": proposal.canonical_model_candidate,
        "proposal_date": proposal.proposal_date.isoformat(),
        "changes": [fact(change) for change in changes],
        "conflicts": [fact(change) for change in proposal.conflicting_facts],
        "unknown_fields": sorted({change.field_path for change in proposal.unsupported_claims}),
        "warnings": proposal.warnings,
        "candidate_errors": proposal.candidate_errors,
    }


def proposal_as_markdown(proposal: RegistryUpdateProposal) -> str:
    candidate_identity = (proposal.candidate_model or {}).get("identity", {})
    title = candidate_identity.get("display_name") or proposal.subject
    unknown_fields = sorted({item.field_path for item in proposal.unsupported_claims})
    lines = [
        f"# {title} Registry Proposal",
        "",
        "## Summary",
        "",
        f"{len(proposal.new_facts)} facts added",
        f"{len(proposal.changed_facts)} facts changed",
        f"{len(proposal.conflicting_facts)} conflicts detected",
        f"{len(unknown_fields)} fields remain unknown",
        "",
        "## High-impact changes",
        "",
    ]
    material_changes = [
        *proposal.new_facts,
        *proposal.changed_facts,
        *proposal.conflicting_facts,
    ]
    high_impact = [change for change in material_changes if change.risk.value == "high"]
    if not high_impact:
        lines.extend(("No high-impact changes identified.", ""))
    for change in high_impact:
        heading = change.field_path.replace("_", " ").title()
        lines.extend(
            (
                f"### {heading}",
                "",
                f"Current: {yaml.safe_dump(change.current_value).strip()}",
                f"Proposed: {yaml.safe_dump(change.proposed_value).strip()}",
                f"Risk: {change.risk.value.upper()}",
                "",
                "Reason:",
                change.reason,
                "",
                "Sources:",
                *[f"- {source}" for source in change.supporting_sources],
                "",
            )
        )
    if unknown_fields:
        lines.extend(("## Unknown fields", "", *[f"- `{path}`" for path in unknown_fields], ""))
    lines.extend(
        (
            "Canonical registry files were not modified. Review this proposal before promotion.",
            "",
        )
    )
    return "\n".join(lines)


def write_proposal_artifacts(
    proposal: RegistryUpdateProposal,
    research: ResearchResult,
    output_directory: Path,
) -> Path:
    """Write review files only when explicitly invoked by an interface."""
    slug = re.sub(r"[^a-z0-9]+", "-", proposal.subject.casefold()).strip("-") or "proposal"
    target = output_directory / slug
    target.mkdir(parents=True, exist_ok=True)
    (target / "proposal.yaml").write_text(proposal_as_yaml(proposal), encoding="utf-8")
    (target / "proposal.md").write_text(proposal_as_markdown(proposal), encoding="utf-8")
    evidence = yaml.safe_dump(research.model_dump(mode="json"), sort_keys=False)
    (target / "evidence.yaml").write_text(evidence, encoding="utf-8")
    if proposal.candidate_model is not None:
        candidate = yaml.safe_dump(proposal.candidate_model, sort_keys=False)
        (target / "candidate-model.yaml").write_text(candidate, encoding="utf-8")
    return target
