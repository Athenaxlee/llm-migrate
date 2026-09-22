from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_migrate.core.agent_research import (
    TrustLevel,
    VerdictKind,
    artifact_sha256,
    build_research_consensus,
    compile_migration_knowledge_candidate,
)
from llm_migrate.core.models import ModelProfile
from llm_migrate.core.proposals import propose_registry_update
from llm_migrate.core.session import (
    SessionOverlayError,
    SessionRegistryManifest,
    build_session_manifest,
    build_session_registry,
    load_session_overlay,
    manifest_summary_lines,
    registry_content_sha256,
    write_session_overlay,
)
from llm_migrate.service import MigrationService
from tests.unit.v11_scenarios import (
    NOW,
    SOURCE_IDENTITY,
    TARGET_IDENTITY,
    pair_research,
    standard_request,
    supportive_review,
    target_research,
)

REQUIREMENTS_SHA = "0" * 64
BASE_SHA = "1" * 64


def _nova_candidate(service: MigrationService) -> ModelProfile:
    research = target_research()
    proposal = propose_registry_update(research, service.registry)
    assert proposal.candidate_model is not None
    return ModelProfile.model_validate(proposal.candidate_model)


def _suitable_consensuses(service: MigrationService):  # type: ignore[no-untyped-def]
    request = standard_request(REQUIREMENTS_SHA)
    outputs = []
    for research, researcher in (
        (target_research(), "researcher-target"),
        (pair_research(), "researcher-pair"),
    ):
        review = supportive_review(
            research,
            reviewer_id="reviewer-independent",
            researcher_id=researcher,
            provider="openai",
        )
        outputs.append(build_research_consensus(research, review, request.topics))
    return outputs


def _manifest(service: MigrationService, **overrides):  # type: ignore[no-untyped-def]
    consensuses = _suitable_consensuses(service)
    candidate = _nova_candidate(service)
    knowledge = compile_migration_knowledge_candidate(
        "session-run-nova-001-migration",
        SOURCE_IDENTITY.model,
        TARGET_IDENTITY.model,
        pair_research(),
        consensuses[1],
    )
    assert knowledge is not None
    manifest = build_session_manifest(
        run_id="run-nova-001",
        created_at=NOW,
        expires_at=NOW + timedelta(days=7),
        base_registry_sha256=BASE_SHA,
        request_sha256=REQUIREMENTS_SHA,
        consensuses=consensuses,
        candidate_profiles=[candidate],
        migration_knowledge=[knowledge],
    )
    if overrides:
        manifest = manifest.model_copy(update=overrides)
    return manifest, [candidate], [knowledge]


def test_manifest_records_hash_links_and_trust(service: MigrationService) -> None:
    manifest, candidates, knowledge = _manifest(service)
    assert manifest.trust_level is TrustLevel.SESSION_AGENT_REVIEWED
    assert manifest.candidate_profiles[0].sha256 == artifact_sha256(candidates[0])
    assert manifest.migration_knowledge[0].sha256 == artifact_sha256(knowledge[0])
    assert {stage.subject for stage in manifest.stage_hashes} == {
        "gpt-6.0-nova",
        "claude-sonnet-5 -> gpt-6.0-nova",
    }
    summary = manifest_summary_lines(manifest)
    assert any("session_agent_reviewed" in line for line in summary)
    assert any("non-canonical" in line for line in summary)


def test_manifest_can_never_claim_canonical_trust() -> None:
    with pytest.raises(ValidationError, match="never carry canonical trust"):
        SessionRegistryManifest(
            run_id="run-x",
            created_at=NOW,
            expires_at=NOW + timedelta(days=1),
            base_registry_sha256=BASE_SHA,
            request_sha256=REQUIREMENTS_SHA,
            trust_level=TrustLevel.CANONICAL_VERIFIED,
        )


def test_overlay_merges_missing_model_without_touching_base(
    service: MigrationService,
) -> None:
    manifest, candidates, knowledge = _manifest(service)
    base_count = len(service.registry.all())
    merged = build_session_registry(service.registry, manifest, candidates, knowledge, as_of=NOW)
    assert len(merged.all()) == base_count + 1
    assert merged.get("gpt-6.0-nova").identity.display_name == "GPT-6.0 Nova"
    assert len(service.registry.all()) == base_count
    with pytest.raises(Exception, match="unknown canonical model"):
        service.registry.get("gpt-6.0-nova")


def test_expired_overlay_is_refused(service: MigrationService) -> None:
    manifest, candidates, knowledge = _manifest(service)
    with pytest.raises(SessionOverlayError, match="expired"):
        build_session_registry(
            service.registry,
            manifest,
            candidates,
            knowledge,
            as_of=NOW + timedelta(days=8),
        )


def test_unreviewed_overlay_is_refused(service: MigrationService) -> None:
    manifest, candidates, knowledge = _manifest(service, trust_level=TrustLevel.SESSION_UNREVIEWED)
    with pytest.raises(SessionOverlayError, match="session_unreviewed"):
        build_session_registry(service.registry, manifest, candidates, knowledge, as_of=NOW)


def test_tampered_candidate_is_refused(service: MigrationService) -> None:
    manifest, candidates, knowledge = _manifest(service)
    tampered = candidates[0].model_copy(
        update={
            "identity": candidates[0].identity.model_copy(
                update={"display_name": "Totally Different"}
            )
        }
    )
    with pytest.raises(SessionOverlayError, match="immutable for the run"):
        build_session_registry(service.registry, manifest, [tampered], knowledge, as_of=NOW)


def test_duplicate_canonical_name_requires_explicit_shadow(
    service: MigrationService,
) -> None:
    manifest, candidates, knowledge = _manifest(service)
    sonnet = service.get_model_profile("claude-sonnet-5")
    shadow_manifest = build_session_manifest(
        run_id="run-shadow-001",
        created_at=NOW,
        expires_at=NOW + timedelta(days=7),
        base_registry_sha256=BASE_SHA,
        request_sha256=REQUIREMENTS_SHA,
        consensuses=_suitable_consensuses(service),
        candidate_profiles=[*candidates, sonnet],
        migration_knowledge=knowledge,
    )
    with pytest.raises(SessionOverlayError, match="explicit selection for each"):
        build_session_registry(
            service.registry,
            shadow_manifest,
            [*candidates, sonnet],
            knowledge,
            as_of=NOW,
        )

    explicit = build_session_manifest(
        run_id="run-shadow-001",
        created_at=NOW,
        expires_at=NOW + timedelta(days=7),
        base_registry_sha256=BASE_SHA,
        request_sha256=REQUIREMENTS_SHA,
        consensuses=_suitable_consensuses(service),
        candidate_profiles=[*candidates, sonnet],
        migration_knowledge=knowledge,
        shadowed_canonical=["claude-sonnet-5"],
    )
    merged = build_session_registry(
        service.registry,
        explicit,
        [*candidates, sonnet],
        knowledge,
        as_of=NOW,
    )
    assert merged.get("claude-sonnet-5") == sonnet


def test_shadow_selection_requires_a_session_candidate() -> None:
    with pytest.raises(ValidationError, match="shadow selections without session candidates"):
        SessionRegistryManifest(
            run_id="run-x",
            created_at=NOW,
            expires_at=NOW + timedelta(days=1),
            base_registry_sha256=BASE_SHA,
            request_sha256=REQUIREMENTS_SHA,
            trust_level=TrustLevel.SESSION_AGENT_REVIEWED,
            shadowed_canonical=["claude-sonnet-5"],
        )


def test_overlay_round_trips_through_run_workspace(
    service: MigrationService, tmp_path: Path
) -> None:
    manifest, candidates, knowledge = _manifest(service)
    run_dir = tmp_path / "runs" / manifest.run_id
    write_session_overlay(run_dir, manifest, candidates, knowledge)
    loaded_manifest, loaded_candidates, loaded_knowledge = load_session_overlay(run_dir)
    assert loaded_manifest == manifest
    assert loaded_candidates == candidates
    assert loaded_knowledge == knowledge

    # tampering with a stored candidate is detected on load + build
    candidate_path = next((run_dir / "candidates").glob("*.yaml"))
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8").replace("GPT-6.0 Nova", "GPT-6.0 Nova Edited"),
        encoding="utf-8",
    )
    _, tampered_candidates, _ = load_session_overlay(run_dir)
    with pytest.raises(SessionOverlayError, match="immutable for the run"):
        build_session_registry(
            service.registry, loaded_manifest, tampered_candidates, loaded_knowledge, as_of=NOW
        )


def test_registry_content_hash_tracks_canonical_changes(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    file = tmp_path / "models" / "a.yaml"
    file.write_text("x: 1\n", encoding="utf-8")
    first = registry_content_sha256(tmp_path)
    assert first == registry_content_sha256(tmp_path)
    file.write_text("x: 2\n", encoding="utf-8")
    assert registry_content_sha256(tmp_path) != first


def test_blocked_topics_and_conflicts_surface_in_manifest(
    service: MigrationService,
) -> None:
    request = standard_request(REQUIREMENTS_SHA)
    research = target_research()
    review = supportive_review(
        research,
        reviewer_id="reviewer-independent",
        researcher_id="researcher-target",
        provider="openai",
    )
    stale = review.model_copy(
        update={
            "verdicts": [
                verdict.model_copy(update={"verdict": VerdictKind.STALE})
                if verdict.claim_id == "nova-price-input"
                else verdict
                for verdict in review.verdicts
            ]
        }
    )
    consensus = build_research_consensus(research, stale, request.topics)
    manifest = build_session_manifest(
        run_id="run-stale-001",
        created_at=NOW,
        expires_at=NOW + timedelta(days=7),
        base_registry_sha256=BASE_SHA,
        request_sha256=REQUIREMENTS_SHA,
        consensuses=[consensus],
        candidate_profiles=[],
        migration_knowledge=[],
    )
    stale_rejection = next(
        item for item in consensus.rejected_claims if item.claim_id == "nova-price-input"
    )
    assert "stale" in stale_rejection.reason
    assert manifest.warnings
    assert any("no accepted claims" in warning for warning in manifest.warnings)
