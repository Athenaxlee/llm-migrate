"""Prompt migration preparation and static validation."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from llm_migrate.analyzers.prompt import analyze_prompt
from llm_migrate.core.models import (
    AdviceBasis,
    MigrationAdvice,
    ModelProfile,
    PromptMigrationSpec,
    PromptSemanticChange,
    PromptValidationResult,
    ResolvedModel,
    SemanticDiffState,
    ValidationIssue,
    ValidationLevel,
)


def _advice(text: str, basis: AdviceBasis = AdviceBasis.HEURISTIC) -> MigrationAdvice:
    return MigrationAdvice(text=text, basis=basis)


def _rewrite_reasoning_instruction(text: str) -> str:
    return re.sub(
        r"(?i)\b(?:think step by step|show your reasoning|show chain[- ]of[- ]thought)\b",
        "Provide a concise justification",
        text,
    )


def prepare_prompt_migration(
    source: ModelProfile | ResolvedModel,
    target: ModelProfile | ResolvedModel,
    prompt: str,
    *,
    source_path: str | None = None,
    source_role: Literal["system", "user", "developer", "unknown"] = "unknown",
) -> PromptMigrationSpec:
    source_profile = source.profile if isinstance(source, ResolvedModel) else source
    target_profile = target.profile if isinstance(target, ResolvedModel) else target
    source_capabilities = (
        source.effective_capabilities if isinstance(source, ResolvedModel) else source.capabilities
    )
    target_capabilities = (
        target.effective_capabilities if isinstance(target, ResolvedModel) else target.capabilities
    )
    source_platform = (
        source.platform.platform if isinstance(source, ResolvedModel) and source.platform else None
    )
    target_platform = (
        target.platform.platform if isinstance(target, ResolvedModel) and target.platform else None
    )
    analysis = analyze_prompt(prompt)
    categories = {finding.category for finding in analysis.findings}
    preserve: list[MigrationAdvice] = []
    if "structured_output" in categories:
        preserve.append(
            _advice(
                "Preserve the required output shape and downstream parser contract.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    if "few_shot_examples" in categories:
        preserve.append(
            _advice(
                "Preserve the behavioral requirements demonstrated by examples.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    if "tool_instructions" in categories:
        preserve.append(
            _advice(
                "Preserve tool-selection constraints and tool result handling semantics.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    guidance_fields = (
        "preferred_structure",
        "delimiter_guidance",
        "reasoning_guidance",
        "verbosity_sensitivity",
        "negative_instruction_guidance",
        "structured_output_guidance",
        "tool_use_guidance",
    )
    source_structure = [
        item for field in guidance_fields for item in getattr(source_profile.prompt_guidance, field)
    ]
    target_structure = [
        item for field in guidance_fields for item in getattr(target_profile.prompt_guidance, field)
    ]
    assumptions = [
        _advice(f"Reconsider source-specific guidance: {item}", AdviceBasis.DETERMINISTIC)
        for item in source_structure
        if item not in target_structure
    ]
    removable = []
    if "duplicated_requirements" in categories:
        removable.append(
            _advice("Review repeated instructions for consolidation after validation.")
        )
    if "assistant_prefill" in categories:
        removable.append(
            _advice(
                "Review the assistant-prefill workaround; target request semantics may not "
                "support it.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    strengthening = [_advice(item, AdviceBasis.DETERMINISTIC) for item in target_structure]
    target_features: list[MigrationAdvice] = []
    if target_capabilities.structured_output and "structured_output" in categories:
        target_features.append(
            _advice(
                "Target native structured output may replace some format-enforcement prose.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    if target_capabilities.tool_use and "tool_instructions" in categories:
        target_features.append(
            _advice(
                "Target native tool schemas may replace duplicated tool syntax in the prompt.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    reasoning = [
        _advice(item, AdviceBasis.DETERMINISTIC)
        for item in target_profile.prompt_guidance.reasoning_guidance
    ]
    if "explicit_chain_of_thought" in categories:
        reasoning.append(
            _advice(
                "Test replacing explicit chain-of-thought requests with "
                "outcome-focused instructions."
            )
        )
    structured = [
        _advice(item, AdviceBasis.DETERMINISTIC)
        for item in target_profile.prompt_guidance.structured_output_guidance
    ]
    risks: list[MigrationAdvice] = []
    for capability in ("structured_output", "tool_use"):
        if getattr(source_capabilities, capability) and not getattr(
            target_capabilities, capability
        ):
            risks.append(
                _advice(
                    f"Target lacks the source model's declared {capability} capability.",
                    AdviceBasis.DETERMINISTIC,
                )
            )
    source_context = source_capabilities.context_window_tokens
    target_context = target_capabilities.context_window_tokens
    if source_context and target_context and target_context < source_context:
        risks.append(
            _advice(
                "Target context window is smaller than the source context window.",
                AdviceBasis.DETERMINISTIC,
            )
        )
    candidate = prompt
    reasoning_revised = False
    if "explicit_chain_of_thought" in categories and target_capabilities.reasoning:
        revised = _rewrite_reasoning_instruction(candidate)
        reasoning_revised = revised != candidate
        candidate = revised
    semantic_diff: list[PromptSemanticChange] = []
    for concern, values in (
        ("objective", analysis.intent.objective),
        ("input/output contract", analysis.intent.input_output_contracts),
        ("examples", analysis.intent.examples),
        ("grounding policy", analysis.intent.grounding_policies),
        ("tool policy", analysis.intent.tool_policies),
        ("verbosity policy", analysis.intent.verbosity_policies),
    ):
        if values:
            before_text = "\n".join(values)
            after_text = (
                _rewrite_reasoning_instruction(before_text) if reasoning_revised else before_text
            )
            semantic_diff.append(
                PromptSemanticChange(
                    state=SemanticDiffState.PRESERVED,
                    concern=concern,
                    before=before_text,
                    after=after_text,
                    rationale="The review candidate retains the detected semantic requirement.",
                    basis=AdviceBasis.DETERMINISTIC,
                )
            )
    if reasoning_revised:
        semantic_diff.append(
            PromptSemanticChange(
                state=SemanticDiffState.REPLACED,
                concern="reasoning policy",
                before="Explicit reasoning-process request",
                after="Provide a concise justification",
                rationale=(
                    "Use an outcome-focused instruction and configure target reasoning "
                    "through supported invocation controls."
                ),
                basis=AdviceBasis.HEURISTIC,
            )
        )
    if "assistant_prefill" in categories:
        semantic_diff.append(
            PromptSemanticChange(
                state=SemanticDiffState.UNRESOLVED,
                concern="assistant-prefill workaround",
                before="Assistant response prefix is controlled by prompt text.",
                rationale="Provider support and equivalent target behavior require review.",
                basis=AdviceBasis.DETERMINISTIC,
            )
        )
    if source_role == "system":
        role_fields = {
            "anthropic-api": "system request field",
            "openai-api": "instructions request field",
            "amazon-bedrock": "system content blocks",
        }
        source_transport = (
            role_fields.get(source_platform, "source system-message mechanism")
            if source_platform
            else "source system-message mechanism"
        )
        target_transport = role_fields.get(target_platform) if target_platform else None
        semantic_diff.append(
            PromptSemanticChange(
                state=(
                    SemanticDiffState.REPLACED
                    if target_transport and target_transport != source_transport
                    else (
                        SemanticDiffState.PRESERVED
                        if target_transport
                        else SemanticDiffState.UNRESOLVED
                    )
                ),
                concern="system-message transport",
                before=source_transport,
                after=target_transport,
                rationale=(
                    "Preserve system-role semantics while moving the content to the target "
                    "request field."
                    if target_transport
                    else "Target system-message transport is unresolved without platform context."
                ),
                basis=AdviceBasis.DETERMINISTIC,
            )
        )
    for item in assumptions:
        semantic_diff.append(
            PromptSemanticChange(
                state=SemanticDiffState.UNRESOLVED,
                concern="source-model prompting assumption",
                before=item.text,
                rationale="Registry guidance differs and requires human review.",
                basis=item.basis,
            )
        )
    for item in strengthening:
        semantic_diff.append(
            PromptSemanticChange(
                state=SemanticDiffState.UNRESOLVED,
                concern="target prompt structure",
                after=item.text,
                rationale=(
                    "Target registry guidance recommends this structure; it was not "
                    "inserted automatically."
                ),
                basis=item.basis,
            )
        )
    if not semantic_diff:
        semantic_diff.append(
            PromptSemanticChange(
                state=SemanticDiffState.PRESERVED,
                concern="prompt text",
                before=prompt,
                after=candidate,
                rationale="No deterministic rewrite was justified.",
                basis=AdviceBasis.DETERMINISTIC,
            )
        )
    return PromptMigrationSpec(
        source_model=source_profile.identity.canonical_name,
        target_model=target_profile.identity.canonical_name,
        source_provider=source_profile.identity.provider,
        target_provider=target_profile.identity.provider,
        source_platform=source_platform,
        target_platform=target_platform,
        source_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        source_path=source_path,
        source_role=source_role,
        source_prompt_analysis=analysis,
        candidate_prompt=candidate,
        semantic_diff=semantic_diff,
        requirements_to_preserve=preserve,
        assumptions_to_reconsider=assumptions,
        instructions_potentially_removable=removable,
        instructions_needing_strengthening=strengthening,
        target_features_replacing_prompt_text=target_features,
        reasoning_configuration=reasoning,
        structured_output=structured,
        token_efficiency_opportunities=(
            [_advice("Consolidate verified duplicate requirements.")]
            if "duplicated_requirements" in categories
            else []
        ),
        hallucination_control=[
            _advice(
                "Retain grounding, citation, and uncertainty requirements and test them explicitly."
            )
        ],
        migration_risks=risks,
        validation_recommendations=[
            _advice(
                "Compare representative task outcomes, formats, tool calls, latency, and token use."
            ),
            _advice("Include edge cases near context and output limits."),
        ],
    )


def validate_prompt(
    target: ResolvedModel,
    prompt: str,
    *,
    source_path: str | None = None,
    target_platform: str | None = None,
) -> PromptValidationResult:
    analysis = analyze_prompt(prompt)
    categories = {item.category for item in analysis.findings}
    issues: list[ValidationIssue] = []
    if not prompt.strip():
        issues.append(
            ValidationIssue(
                code="empty_prompt",
                level=ValidationLevel.BLOCKER,
                message="Prompt is empty.",
                source_path=source_path,
            )
        )
    target_capabilities = target.effective_capabilities
    target_context = target_capabilities.context_window_tokens
    if target_context is not None and analysis.approximate_token_count > target_context:
        issues.append(
            ValidationIssue(
                code="context_window_exceeded",
                level=ValidationLevel.BLOCKER,
                message="Approximate prompt size exceeds the target context window.",
                source_path=source_path,
            )
        )
    elif target_context is None and prompt.strip():
        issues.append(
            ValidationIssue(
                code="unknown_context_window",
                level=ValidationLevel.WARNING,
                message="Target context-window capacity is unknown.",
                source_path=source_path,
            )
        )
    for category, capability, label in (
        ("structured_output", target_capabilities.structured_output, "structured output"),
        ("tool_instructions", target_capabilities.tool_use, "tool use"),
    ):
        if category not in categories:
            continue
        if capability is False:
            issues.append(
                ValidationIssue(
                    code=f"unsupported_{category}",
                    level=ValidationLevel.BLOCKER,
                    message=f"Prompt assumes {label}, but the target declares it unsupported.",
                    source_path=source_path,
                )
            )
        elif capability is None:
            issues.append(
                ValidationIssue(
                    code=f"unknown_{category}",
                    level=ValidationLevel.WARNING,
                    message=f"Prompt assumes {label}; target support is unknown.",
                    source_path=source_path,
                )
            )
    if "explicit_chain_of_thought" in categories:
        issues.append(
            ValidationIssue(
                code="reasoning_process_instruction",
                level=ValidationLevel.WARNING,
                message="Review explicit reasoning-process instructions for the target model.",
                source_path=source_path,
            )
        )
    if target_platform and not any(item.platform == target_platform for item in target.platforms):
        issues.append(
            ValidationIssue(
                code="invalid_target_platform",
                level=ValidationLevel.BLOCKER,
                message=f"Target model has no {target_platform} representation.",
                source_path=source_path,
            )
        )
    return PromptValidationResult(
        valid=not any(item.level is ValidationLevel.BLOCKER for item in issues),
        source_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        target_model=target.identity.canonical_name,
        target_platform=target_platform,
        issues=issues,
    )
