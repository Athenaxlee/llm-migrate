from __future__ import annotations

import json
from pathlib import Path

from llm_migrate.analyzers.prompt import analyze_prompt
from llm_migrate.core.models import AdviceBasis, SemanticDiffState, ValidationLevel
from llm_migrate.service import MigrationService

PROMPT = """# Task
Think step by step and return JSON.
Never invent facts. Do not omit fields. Avoid prose. You must not add keys.
Use the weather tool when needed.
Example 1:
Input: Seattle
Output: {"city": "Seattle"}
Use the weather tool when needed.
<context>Grounded data</context>
"""


def test_static_prompt_heuristics_find_characteristics() -> None:
    result = analyze_prompt(PROMPT)
    categories = {finding.category for finding in result.findings}
    assert {
        "duplicated_requirements",
        "negative_wording",
        "explicit_chain_of_thought",
        "structured_output",
        "few_shot_examples",
        "tool_instructions",
        "section_structure",
        "provider_specific_formatting",
    } <= categories
    assert result.approximate_token_count == (len(PROMPT) + 3) // 4
    assert result.caveat.startswith("Static heuristics")


def test_prompt_migration_spec_preserves_contract_without_rewriting(
    service: MigrationService,
) -> None:
    result = service.prepare_prompt_migration(
        "alpha large",
        "beta balanced",
        PROMPT,
        source_path="prompts/system.txt",
        source_role="system",
    )
    assert result.source_prompt_analysis.character_count == len(PROMPT)
    assert any("output shape" in item.text for item in result.requirements_to_preserve)
    assert any(item.basis is AdviceBasis.DETERMINISTIC for item in result.reasoning_configuration)
    # Minimal adaptation: the deterministic candidate is the source prompt
    # verbatim; recommended changes are carried as advice, never auto-applied.
    assert result.candidate_prompt == PROMPT
    assert result.source_prompt_sha256
    assert result.source_path == "prompts/system.txt"
    assert result.source_role == "system"
    assert all(item.state is not SemanticDiffState.REPLACED for item in result.semantic_diff)
    assert any("chain-of-thought" in item.text for item in result.reasoning_configuration)


def test_prompt_validation_reports_target_blockers(service: MigrationService) -> None:
    result = service.validate_prompt("gamma cheap", "Use the weather tool. Return JSON.")
    assert not result.valid
    assert {item.code for item in result.issues} == {
        "unsupported_structured_output",
        "unsupported_tool_instructions",
    }
    assert all(item.level is ValidationLevel.BLOCKER for item in result.issues)


def test_prompt_validation_returns_platform_errors_as_blockers(
    service: MigrationService,
) -> None:
    invalid = service.validate_prompt(
        "gamma cheap", "Answer briefly.", target_platform="openai-api"
    )
    ambiguous = service.validate_prompt(
        "claude-sonnet-5", "Answer briefly.", target_platform="amazon-bedrock"
    )
    assert not invalid.valid
    assert any(item.code == "invalid_target_platform" for item in invalid.issues)
    assert not ambiguous.valid
    assert any(item.code == "ambiguous_target_platform" for item in ambiguous.issues)


def test_system_prompt_transport_and_provider_guidance_are_explicit(
    service: MigrationService,
) -> None:
    result = service.prepare_prompt_migration(
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "<task>Return JSON only.</task>",
        source_role="system",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert result.source_platform == "anthropic-api"
    assert result.target_platform == "openai-api"
    role_change = next(
        item for item in result.semantic_diff if item.concern == "system-message transport"
    )
    assert role_change.state is SemanticDiffState.REPLACED
    assert role_change.before == "system request field"
    assert role_change.after == "instructions request field"
    assert any(
        item.category == "json_only_prompting" for item in result.source_prompt_analysis.findings
    )


def test_deprecated_assistant_prefill_is_left_for_review(service: MigrationService) -> None:
    result = service.prepare_prompt_migration(
        "claude-sonnet-5",
        "gpt-5.6-sol",
        "Begin your response with { and return JSON only.",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    assert any(
        item.category == "assistant_prefill" for item in result.source_prompt_analysis.findings
    )
    assert any(
        item.concern == "assistant-prefill workaround"
        and item.state is SemanticDiffState.UNRESOLVED
        for item in result.semantic_diff
    )


def test_cross_provider_prompt_review_surface_matches_golden(
    service: MigrationService, project_root: Path
) -> None:
    result = service.prepare_prompt_migration(
        "claude-sonnet-5",
        "gpt-5.6-sol",
        PROMPT,
        source_path="prompts/system.txt",
        source_role="system",
        source_platform="anthropic-api",
        target_platform="openai-api",
    )
    review_surface = {
        "source_model": result.source_model,
        "target_model": result.target_model,
        "source_provider": result.source_provider,
        "target_provider": result.target_provider,
        "source_platform": result.source_platform,
        "target_platform": result.target_platform,
        "source_role": result.source_role,
        "candidate_prompt": result.candidate_prompt,
        "semantic_diff": [
            {
                "state": item.state.value,
                "concern": item.concern,
                "before": item.before,
                "after": item.after,
            }
            for item in result.semantic_diff
        ],
        "finding_categories": [item.category for item in result.source_prompt_analysis.findings],
    }
    expected = json.loads(
        (project_root / "tests" / "golden" / "v03" / "anthropic_to_openai_prompt.json").read_text()
    )
    assert review_surface == expected
