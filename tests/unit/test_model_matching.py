"""Lenient, registry-first identifier matching (V1.2)."""

from __future__ import annotations

from llm_migrate.core.models import ModelMatchStatus
from llm_migrate.core.resolver import strip_identifier_decorations
from llm_migrate.service import MigrationService


def test_exact_identifier_still_resolves(service: MigrationService) -> None:
    match = service.match_model("claude-sonnet-5", "anthropic-api")
    assert match.status is ModelMatchStatus.RESOLVED
    assert match.canonical_name == "claude-sonnet-5"
    assert match.model_id == "claude-sonnet-5"


def test_bedrock_inference_profile_prefix_resolves_deterministically(
    service: MigrationService,
) -> None:
    match = service.match_model("us.anthropic.claude-sonnet-4-6", "amazon-bedrock")
    assert match.status is ModelMatchStatus.RESOLVED
    assert match.canonical_name == "claude-sonnet-4-6"
    assert match.model_id == "anthropic.claude-sonnet-4-6"
    assert any("inference-profile" in note for note in match.notes)


def test_version_suffix_and_prefix_both_strip() -> None:
    assert (
        strip_identifier_decorations("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        == "anthropic.claude-sonnet-4-5-20250929"
    )
    assert strip_identifier_decorations("eu.us.global.model") == "model"


def test_vague_platform_name_is_interpreted(service: MigrationService) -> None:
    match = service.match_model("us.anthropic.claude-sonnet-4-6", "bedrock")
    assert match.status is ModelMatchStatus.RESOLVED
    assert match.platform == "amazon-bedrock"
    assert any("Interpreted platform" in note for note in match.notes)


def test_misspelled_identifier_returns_ranked_candidates(service: MigrationService) -> None:
    match = service.match_model("claude sonet 5")
    assert match.status is ModelMatchStatus.NEEDS_CONFIRMATION
    assert match.candidates[0].canonical_name == "claude-sonnet-5"
    assert match.candidates == sorted(
        match.candidates, key=lambda item: (-item.similarity, item.canonical_name)
    )
    assert "Confirm" in match.guidance


def test_multiple_platform_representations_need_confirmation(
    service: MigrationService,
) -> None:
    match = service.match_model("us.anthropic.claude-sonnet-5", "amazon-bedrock")
    assert match.status is ModelMatchStatus.NEEDS_CONFIRMATION
    endpoints = {candidate.endpoint for candidate in match.candidates}
    assert endpoints == {"bedrock-runtime", "bedrock-mantle"}
    assert all(candidate.canonical_name == "claude-sonnet-5" for candidate in match.candidates)


def test_endpoint_context_resolves_the_ambiguity(service: MigrationService) -> None:
    match = service.match_model("us.anthropic.claude-sonnet-5", "amazon-bedrock", "bedrock-runtime")
    assert match.status is ModelMatchStatus.RESOLVED
    assert match.canonical_name == "claude-sonnet-5"


def test_unknown_identifier_is_not_found_with_research_guidance(
    service: MigrationService,
) -> None:
    match = service.match_model("entirely-unrelated-model-xyz")
    assert match.status is ModelMatchStatus.NOT_FOUND
    assert not match.candidates
    assert "research" in match.guidance


def test_wrong_platform_lists_real_representations(service: MigrationService) -> None:
    match = service.match_model("claude-sonnet-4-6", "openai-api")
    assert match.status is ModelMatchStatus.NEEDS_CONFIRMATION
    platforms = {candidate.platform for candidate in match.candidates}
    assert platforms == {"anthropic-api", "amazon-bedrock"}
