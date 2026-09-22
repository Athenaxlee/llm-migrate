"""V1.1 immutable, expiring, user-scoped session registry overlays.

A session overlay supplements — or, only by explicit selection, shadows — the
canonical registry for one recorded run. It is content-addressed, visibly
non-canonical, and never touches checked-in `registry/` files.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from llm_migrate.core.agent_research import (
    RUN_ID_PATTERN,
    SHA256_PATTERN,
    ResearchConsensus,
    ResearchTopic,
    SessionSuitability,
    TrustLevel,
    artifact_sha256,
)
from llm_migrate.core.knowledge import MigrationKnowledge
from llm_migrate.core.models import ModelProfile, StrictModel
from llm_migrate.core.registry import ModelRegistry, RegistryError
from llm_migrate.core.runstate import atomic_write_text


class SessionOverlayError(RegistryError):
    """A session overlay is invalid, expired, tampered with, or unauthorized."""


class SessionArtifactRef(StrictModel):
    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)


class SessionStageHashes(StrictModel):
    subject: str = Field(min_length=1)
    research_sha256: str = Field(pattern=SHA256_PATTERN)
    review_sha256: str = Field(pattern=SHA256_PATTERN)
    consensus_sha256: str = Field(pattern=SHA256_PATTERN)


class SessionRegistryManifest(StrictModel):
    """Immutable record of what one run's overlay contains and may claim."""

    schema_version: Literal["1"] = "1"
    run_id: str = Field(min_length=1, pattern=RUN_ID_PATTERN)
    created_at: datetime
    expires_at: datetime
    base_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    stage_hashes: list[SessionStageHashes] = Field(default_factory=list)
    trust_level: TrustLevel
    candidate_profiles: list[SessionArtifactRef] = Field(default_factory=list)
    migration_knowledge: list[SessionArtifactRef] = Field(default_factory=list)
    shadowed_canonical: list[str] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    blocked_topics: list[ResearchTopic] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest(self) -> SessionRegistryManifest:
        if self.expires_at <= self.created_at:
            raise ValueError("session overlay must expire after it is created")
        if self.trust_level is TrustLevel.CANONICAL_VERIFIED:
            raise ValueError("a session overlay can never carry canonical trust")
        candidate_names = [ref.name for ref in self.candidate_profiles]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("duplicate session candidate profile names")
        knowledge_names = [ref.name for ref in self.migration_knowledge]
        if len(knowledge_names) != len(set(knowledge_names)):
            raise ValueError("duplicate session migration knowledge IDs")
        unknown_shadows = set(self.shadowed_canonical) - set(candidate_names)
        if unknown_shadows:
            raise ValueError(
                f"shadow selections without session candidates: {sorted(unknown_shadows)}"
            )
        return self


def registry_content_sha256(registry_root: Path | str) -> str:
    """Hash the canonical registry files an overlay was built against."""
    root = Path(registry_root)
    digest = hashlib.sha256()
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".yaml", ".yml"}
    )
    if not paths:
        raise SessionOverlayError(f"no registry files found under: {root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def build_session_manifest(
    *,
    run_id: str,
    created_at: datetime,
    expires_at: datetime,
    base_registry_sha256: str,
    request_sha256: str,
    consensuses: list[ResearchConsensus],
    candidate_profiles: list[ModelProfile],
    migration_knowledge: list[MigrationKnowledge],
    shadowed_canonical: list[str] | None = None,
) -> SessionRegistryManifest:
    """Deterministically derive the manifest from validated stage artifacts."""
    if any(item.suitability is not SessionSuitability.SUITABLE for item in consensuses):
        trust = TrustLevel.SESSION_UNREVIEWED
    elif consensuses:
        trust = TrustLevel.SESSION_AGENT_REVIEWED
    else:
        trust = TrustLevel.SESSION_UNREVIEWED
    unresolved = sorted(
        {
            f"{conflict.field_path}: {conflict.statement}"
            for consensus in consensuses
            for conflict in consensus.unresolved_conflicts
        }
    )
    blocked = sorted(
        {
            topic
            for consensus in consensuses
            for topic, coverage in consensus.topic_coverage.items()
            if coverage.value == "blocked"
        },
        key=lambda topic: topic.value,
    )
    warnings = sorted({warning for consensus in consensuses for warning in consensus.warnings})
    return SessionRegistryManifest(
        run_id=run_id,
        created_at=created_at,
        expires_at=expires_at,
        base_registry_sha256=base_registry_sha256,
        request_sha256=request_sha256,
        stage_hashes=[
            SessionStageHashes(
                subject=consensus.subject,
                research_sha256=consensus.research_sha256,
                review_sha256=consensus.review_sha256,
                consensus_sha256=artifact_sha256(consensus),
            )
            for consensus in sorted(consensuses, key=lambda item: item.subject)
        ],
        trust_level=trust,
        candidate_profiles=[
            SessionArtifactRef(
                name=profile.identity.canonical_name,
                sha256=artifact_sha256(profile),
            )
            for profile in sorted(candidate_profiles, key=lambda item: item.identity.canonical_name)
        ],
        migration_knowledge=[
            SessionArtifactRef(name=item.id, sha256=artifact_sha256(item))
            for item in sorted(migration_knowledge, key=lambda item: item.id)
        ],
        shadowed_canonical=sorted(shadowed_canonical or []),
        unresolved_conflicts=unresolved,
        blocked_topics=blocked,
        warnings=warnings,
    )


def build_session_registry(
    base: ModelRegistry,
    manifest: SessionRegistryManifest,
    candidate_profiles: list[ModelProfile],
    migration_knowledge: list[MigrationKnowledge],
    *,
    as_of: datetime,
) -> ModelRegistry:
    """Merge session knowledge over the canonical registry, in memory only.

    Consequential use requires `session_agent_reviewed` trust, an unexpired
    overlay, and content hashes that still match the manifest. Session
    candidates may shadow a canonical profile only through the manifest's
    explicit `shadowed_canonical` selection.
    """
    if manifest.trust_level is not TrustLevel.SESSION_AGENT_REVIEWED:
        raise SessionOverlayError(
            "session_unreviewed knowledge cannot support consequential migration "
            "operations; only session_agent_reviewed overlays may be loaded"
        )
    if as_of >= manifest.expires_at:
        raise SessionOverlayError(
            f"session overlay for run {manifest.run_id!r} expired at "
            f"{manifest.expires_at.isoformat()}"
        )
    expected_profiles = {ref.name: ref.sha256 for ref in manifest.candidate_profiles}
    provided_profiles = {profile.identity.canonical_name: profile for profile in candidate_profiles}
    if set(expected_profiles) != set(provided_profiles):
        raise SessionOverlayError(
            "session candidate profiles do not match the manifest: expected "
            f"{sorted(expected_profiles)}, got {sorted(provided_profiles)}"
        )
    for name, profile in provided_profiles.items():
        if artifact_sha256(profile) != expected_profiles[name]:
            raise SessionOverlayError(
                f"session candidate {name!r} no longer matches its manifest hash; "
                "overlays are immutable for the run"
            )
    expected_knowledge = {ref.name: ref.sha256 for ref in manifest.migration_knowledge}
    provided_knowledge = {item.id: item for item in migration_knowledge}
    if set(expected_knowledge) != set(provided_knowledge):
        raise SessionOverlayError(
            "session migration knowledge does not match the manifest: expected "
            f"{sorted(expected_knowledge)}, got {sorted(provided_knowledge)}"
        )
    for knowledge_id, item in provided_knowledge.items():
        if artifact_sha256(item) != expected_knowledge[knowledge_id]:
            raise SessionOverlayError(
                f"session migration knowledge {knowledge_id!r} no longer matches "
                "its manifest hash; overlays are immutable for the run"
            )

    base_names = {profile.identity.canonical_name for profile in base.all()}
    shadows = set(manifest.shadowed_canonical)
    merged_profiles: list[ModelProfile] = []
    for profile in base.all():
        name = profile.identity.canonical_name
        if name in shadows:
            merged_profiles.append(provided_profiles[name])
        else:
            merged_profiles.append(profile)
    # Every canonical-shadow choice surfaces together in ONE error, so the
    # host resolves them in a single pass instead of one failure per retry.
    needing_selection = sorted(
        name for name in provided_profiles if name in base_names and name not in shadows
    )
    if needing_selection:
        raise SessionOverlayError(
            "session candidate(s) duplicate canonical profiles; shadowing requires an "
            "explicit selection for each (pass shadow_canonical): " + ", ".join(needing_selection)
        )
    for name in sorted(provided_profiles):
        if name in base_names:
            continue
        merged_profiles.append(provided_profiles[name])

    base_knowledge_ids = {item.id for item in base.migrations()}
    duplicate_knowledge = base_knowledge_ids & set(provided_knowledge)
    if duplicate_knowledge:
        raise SessionOverlayError(
            "session migration knowledge duplicates canonical knowledge: "
            f"{sorted(duplicate_knowledge)}"
        )
    merged_knowledge = [*base.migrations(), *provided_knowledge.values()]
    return ModelRegistry(merged_profiles, merged_knowledge, base.observations())


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_PATTERN.sub("-", name.casefold()).strip("-") or "artifact"


def write_session_overlay(
    run_dir: Path,
    manifest: SessionRegistryManifest,
    candidate_profiles: list[ModelProfile],
    migration_knowledge: list[MigrationKnowledge],
) -> Path:
    """Persist the overlay beneath the user-scoped run workspace."""
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = run_dir / "candidates"
    knowledge_dir = run_dir / "knowledge"
    for profile in candidate_profiles:
        path = candidates_dir / f"{_slug(profile.identity.canonical_name)}.yaml"
        atomic_write_text(path, yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False))
    for item in migration_knowledge:
        path = knowledge_dir / f"{_slug(item.id)}.yaml"
        atomic_write_text(path, yaml.safe_dump(item.model_dump(mode="json"), sort_keys=False))
    manifest_path = run_dir / "session-manifest.yaml"
    atomic_write_text(
        manifest_path, yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False)
    )
    return manifest_path


def load_session_overlay(
    run_dir: Path,
) -> tuple[SessionRegistryManifest, list[ModelProfile], list[MigrationKnowledge]]:
    """Load and re-validate a persisted overlay without trusting file names."""
    manifest_path = Path(run_dir) / "session-manifest.yaml"
    if not manifest_path.is_file():
        raise SessionOverlayError(f"no session manifest found at: {manifest_path}")
    try:
        manifest = SessionRegistryManifest.model_validate(
            yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        )
        candidates = [
            ModelProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in sorted((Path(run_dir) / "candidates").glob("*.yaml"))
        ]
        knowledge = [
            MigrationKnowledge.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            for path in sorted((Path(run_dir) / "knowledge").glob("*.yaml"))
        ]
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise SessionOverlayError(f"invalid session overlay under {run_dir}: {exc}") from exc
    return manifest, candidates, knowledge


def manifest_summary_lines(manifest: SessionRegistryManifest) -> list[str]:
    """Human-visible provenance for plans and reports that used the overlay."""
    lines = [
        f"Session overlay run: {manifest.run_id}",
        f"Trust level: {manifest.trust_level.value} (non-canonical, user-scoped)",
        f"Created: {manifest.created_at.isoformat()}",
        f"Expires: {manifest.expires_at.isoformat()}",
        f"Base registry sha256: {manifest.base_registry_sha256}",
    ]
    if manifest.candidate_profiles:
        names = ", ".join(ref.name for ref in manifest.candidate_profiles)
        lines.append(f"Session model profiles: {names}")
    if manifest.migration_knowledge:
        names = ", ".join(ref.name for ref in manifest.migration_knowledge)
        lines.append(f"Session migration knowledge: {names}")
    if manifest.shadowed_canonical:
        lines.append(
            "Explicitly shadowed canonical profiles: " + ", ".join(manifest.shadowed_canonical)
        )
    for conflict in manifest.unresolved_conflicts:
        lines.append(f"Unresolved conflict: {conflict}")
    for topic in manifest.blocked_topics:
        lines.append(f"Blocked topic: {topic.value}")
    for warning in manifest.warnings:
        lines.append(f"Warning: {warning}")
    return lines
