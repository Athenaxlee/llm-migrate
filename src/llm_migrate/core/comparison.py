"""Typed, platform-aware deterministic model comparison."""

from __future__ import annotations

from typing import Any

from llm_migrate.core.models import (
    ComparisonSeverity,
    ComparisonState,
    ModelComparison,
    ModelDifference,
    ModelProfile,
    ParameterState,
    PlatformAvailability,
    SourceReference,
)
from llm_migrate.core.resolver import effective_capabilities

_SEVERITY_ORDER = {item: index for index, item in enumerate(ComparisonSeverity)}


def evidence_url(
    profile: ModelProfile,
    section_sources: list[SourceReference],
    topic: str,
) -> str | None:
    """URL backing one side of a claim: field-level sources first, then the
    profile source that declares support for the claim's topic."""
    for source in section_sources:
        if source.url is not None:
            return str(source.url)
    for source in profile.sources:
        if topic in source.supports and source.url is not None:
            return str(source.url)
    return None


def _override_aware_capability_url(
    platform: PlatformAvailability | None,
    field: str,
    profile_url: str | None,
) -> str | None:
    """Evidence for one effective-capability claim.

    A value produced by a platform capability override is a platform-level
    fact: it may only cite the platform entry's own sources. Falling back to
    profile-level documents would hyperlink the claim to a page that can
    assert the opposite value; no link is more honest than a wrong one.
    """
    overrides = platform.capability_overrides if platform else None
    if overrides is not None and getattr(overrides, field) is not None:
        assert platform is not None
        return next((str(source.url) for source in platform.sources if source.url), None)
    return profile_url


def capability_evidence_url(
    profile: ModelProfile,
    platform: PlatformAvailability | None,
    field: str,
) -> str | None:
    """Override-aware evidence URL for one effective-capability fact."""
    capabilities = effective_capabilities(profile, platform)
    profile_url = evidence_url(profile, capabilities.sources, "capabilities")
    return _override_aware_capability_url(platform, field, profile_url)


def _item(
    category: str,
    field: str,
    source: Any,
    target: Any,
    impact: str,
    *,
    loss_is_breaking: bool = False,
    changed_severity: ComparisonSeverity = ComparisonSeverity.MEDIUM,
    action: str | None = None,
    source_evidence_url: str | None = None,
    target_evidence_url: str | None = None,
) -> ModelDifference:
    if source is None or target is None:
        state = ComparisonState.UNKNOWN
        severity = (
            ComparisonSeverity.HIGH
            if source not in (None, False) and target is None
            else ComparisonSeverity.LOW
        )
        impact = f"One or both values for {field!r} are unknown; compatibility must not be assumed."
        action = action or "Verify both source reliance and target support before migration."
    elif source == target:
        state = ComparisonState.SAME
        severity = ComparisonSeverity.INFO
        impact = "No migration-relevant change."
        action = None
    elif target is False or target == ParameterState.UNSUPPORTED:
        state = ComparisonState.UNSUPPORTED
        severity = ComparisonSeverity.BREAKING if loss_is_breaking else ComparisonSeverity.HIGH
    else:
        state = ComparisonState.DIFFERENT
        severity = changed_severity
    return ModelDifference(
        category=category,
        field=field,
        source_value=source,
        target_value=target,
        state=state,
        severity=severity,
        migration_impact=impact,
        recommended_action=action,
        source_evidence_url=source_evidence_url,
        target_evidence_url=target_evidence_url,
    )


def _pricing_sources(profile: ModelProfile) -> list[SourceReference]:
    if profile.pricing is None:
        return []
    components = (
        profile.pricing.input,
        profile.pricing.output,
        profile.pricing.cached_input,
        profile.pricing.batch,
    )
    return [source for component in components if component for source in component.sources]


def compare_models(
    source: ModelProfile,
    target: ModelProfile,
    source_platform: PlatformAvailability | None = None,
    target_platform: PlatformAvailability | None = None,
) -> ModelComparison:
    """Compare the complete V0.2 migration surface, including equal and unknown facts."""
    items: list[ModelDifference] = []
    items.append(
        _item(
            "identity",
            "provider",
            source.identity.provider,
            target.identity.provider,
            "Provider SDK and invocation code may change.",
            changed_severity=ComparisonSeverity.HIGH,
            action="Review authentication, SDK, request, and response mappings.",
            source_evidence_url=evidence_url(source, [], "identity"),
            target_evidence_url=evidence_url(target, [], "identity"),
        )
    )
    items.append(
        _item(
            "platform",
            "representation",
            (
                {
                    "platform": source_platform.platform,
                    "endpoint": source_platform.endpoint,
                    "model_id": source_platform.model_id,
                }
                if source_platform
                else None
            ),
            (
                {
                    "platform": target_platform.platform,
                    "endpoint": target_platform.endpoint,
                    "model_id": target_platform.model_id,
                }
                if target_platform
                else None
            ),
            "Selected deployment platforms differ.",
            changed_severity=ComparisonSeverity.HIGH,
            action="Confirm endpoint, region, authentication, and platform feature support.",
            source_evidence_url=evidence_url(
                source, source_platform.sources if source_platform else [], "platforms"
            ),
            target_evidence_url=evidence_url(
                target, target_platform.sources if target_platform else [], "platforms"
            ),
        )
    )
    items.append(
        _item(
            "lifecycle",
            "status",
            source.lifecycle.status,
            target.lifecycle.status,
            "Operational support horizons differ.",
            source_evidence_url=evidence_url(source, source.lifecycle.sources, "lifecycle"),
            target_evidence_url=evidence_url(target, target.lifecycle.sources, "lifecycle"),
        )
    )

    source_caps = effective_capabilities(source, source_platform)
    target_caps = effective_capabilities(target, target_platform)
    source_caps_url = evidence_url(source, source_caps.sources, "capabilities")
    target_caps_url = evidence_url(target, target_caps.sources, "capabilities")
    for field in (
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
    ):
        old = getattr(source_caps, field)
        new = getattr(target_caps, field)
        category = "multimodality" if field in {"image_input", "document_input"} else field
        items.append(
            _item(
                category,
                field,
                old,
                new,
                f"Capability {field!r} changes.",
                loss_is_breaking=old is True,
                changed_severity=ComparisonSeverity.INFO,
                action=f"Remove or replace reliance on {field}." if old is True else None,
                source_evidence_url=_override_aware_capability_url(
                    source_platform, field, source_caps_url
                ),
                target_evidence_url=_override_aware_capability_url(
                    target_platform, field, target_caps_url
                ),
            )
        )
    for field, label in (
        ("context_window_tokens", "context window"),
        ("maximum_output_tokens", "maximum output"),
    ):
        old = getattr(source_caps, field)
        new = getattr(target_caps, field)
        items.append(
            _item(
                "context_output_limits",
                field,
                old,
                new,
                f"The {label} limit changes.",
                changed_severity=(
                    ComparisonSeverity.HIGH
                    if old is not None and new is not None and new < old
                    else ComparisonSeverity.INFO
                ),
                action="Test workloads near the source limit.",
                source_evidence_url=_override_aware_capability_url(
                    source_platform, field, source_caps_url
                ),
                target_evidence_url=_override_aware_capability_url(
                    target_platform, field, target_caps_url
                ),
            )
        )
    for name in sorted(source.parameters.keys() | target.parameters.keys()):
        old = source.parameters.get(name)
        new = target.parameters.get(name)
        old_state = old.state if old else None
        new_state = new.state if new else None
        items.append(
            _item(
                "parameters",
                name,
                old_state,
                new_state,
                f"Parameter {name!r} support or semantics differ.",
                loss_is_breaking=old_state in {ParameterState.SUPPORTED, ParameterState.REQUIRED},
                action=f"Map, verify, or remove {name!r} before invoking the target.",
                source_evidence_url=evidence_url(source, old.sources if old else [], "parameters"),
                target_evidence_url=evidence_url(target, new.sources if new else [], "parameters"),
            )
        )
    items.append(
        _item(
            "pricing",
            "published_pricing",
            source.pricing.model_dump(mode="json") if source.pricing else None,
            target.pricing.model_dump(mode="json") if target.pricing else None,
            "Token costs may change; workload-specific cost depends on token mix.",
            action="Estimate cost with representative input, output, cache, and batch usage.",
            source_evidence_url=evidence_url(source, _pricing_sources(source), "pricing"),
            target_evidence_url=evidence_url(target, _pricing_sources(target), "pricing"),
        )
    )
    items.append(
        _item(
            "behavioral_prompt_guidance",
            "prompt_guidance",
            source.prompt_guidance.model_dump(mode="json"),
            target.prompt_guidance.model_dump(mode="json"),
            "Evidence-backed prompting guidance differs.",
            action="Revisit registry-backed guidance and validate behavior.",
            source_evidence_url=evidence_url(
                source, source.prompt_guidance.sources, "prompt_guidance"
            ),
            target_evidence_url=evidence_url(
                target, target.prompt_guidance.sources, "prompt_guidance"
            ),
        )
    )
    highest = max(
        (item.severity for item in items),
        key=lambda severity: _SEVERITY_ORDER[severity],
        default=ComparisonSeverity.INFO,
    )
    return ModelComparison(
        source_model=source.identity.canonical_name,
        target_model=target.identity.canonical_name,
        source_platform=source_platform.platform if source_platform else None,
        target_platform=target_platform.platform if target_platform else None,
        differences=items,
        highest_severity=highest,
    )
