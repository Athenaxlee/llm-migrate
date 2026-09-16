"""Application-level V0.4 migration planning and reporting."""

from __future__ import annotations

from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from llm_migrate.core.migration import prepare_prompt_migration, validate_prompt
from llm_migrate.core.models import (
    ApplicationAnalysis,
    ComparisonSeverity,
    CompatibilityState,
    CouplingKind,
    InvocationMigrationSpec,
    MigrationApplicationSummary,
    MigrationEndpoint,
    MigrationPlan,
    ModelComparison,
    PlannedMigrationChange,
    PromptComponent,
    PromptDiscoveryCoverage,
    PromptMigrationSpec,
    PromptSource,
    PromptSourceConfidence,
    ResolvedModel,
    SourceLocation,
    ValidationIssue,
    ValidationLevel,
)


def _endpoint(model: ResolvedModel) -> MigrationEndpoint:
    if model.platform is None:
        raise ValueError(
            f"A platform is required for application migration planning: {model.canonical_name}."
        )
    return MigrationEndpoint(
        model=model.canonical_name,
        provider=model.identity.provider,
        platform=model.platform.platform,
        model_id=model.platform.model_id,
    )


def _locations(application: ApplicationAnalysis, *kinds: CouplingKind) -> list[SourceLocation]:
    accepted = set(kinds)
    return sorted(
        {item.location for item in application.findings if item.kind in accepted},
        key=lambda item: (item.path, item.line, item.column),
    )


def _change(
    category: str, description: str, locations: list[SourceLocation]
) -> PlannedMigrationChange:
    return PlannedMigrationChange.model_validate(
        {
            "category": category,
            "description": description,
            "files": sorted({item.path for item in locations}),
            "locations": locations,
        }
    )


def _prompt_inputs(
    application: ApplicationAnalysis,
) -> tuple[list[tuple[PromptSource, PromptComponent]], list[str]]:
    """Prompt components to prepare, plus unknowns for what stays unresolved."""
    inputs: list[tuple[PromptSource, PromptComponent]] = []
    unknowns: list[str] = []
    for source in application.prompt_sources:
        if source.confidence is PromptSourceConfidence.LOW:
            unknowns.append(
                f"{source.path} contains prompt-like keys but nothing references it; it was "
                "not prepared automatically. Name it in prompt_sources to include it."
            )
            continue
        inputs.extend((source, component) for component in source.components)
    discovery = application.prompt_discovery
    if discovery.dynamic_consumers:
        unknowns.append(
            f"{discovery.dynamic_consumers} prompt consumer(s) supply dynamically built "
            "content with no statically resolvable prompt source; review them manually."
        )
    return inputs, unknowns


def _schema_validation(application: ApplicationAnalysis) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    contracts: list[tuple[str, dict[str, Any] | None, SourceLocation]] = [
        (f"tool {item.name or '<unknown>'}", item.input_schema, item.location)
        for item in application.tool_definitions
    ]
    contracts.extend(
        (f"output {item.name or '<unnamed>'}", item.json_schema, item.location)
        for item in application.structured_outputs
    )
    for label, schema, location in contracts:
        if schema is None:
            issues.append(
                ValidationIssue(
                    code="unresolved_json_schema",
                    level=ValidationLevel.WARNING,
                    message=f"The {label} JSON Schema could not be resolved statically.",
                    source_path=location.path,
                )
            )
            continue
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            issues.append(
                ValidationIssue(
                    code="invalid_json_schema",
                    level=ValidationLevel.BLOCKER,
                    message=f"The {label} JSON Schema is invalid: {exc.message}.",
                    source_path=location.path,
                )
            )
    return issues


def _required_tests(application: ApplicationAnalysis) -> list[str]:
    kinds = {item.kind for item in application.findings}
    tests = ["Run existing unit and integration tests against the target configuration."]
    if CouplingKind.PROMPT in kinds:
        tests.append("Review migrated prompt intent and output behavior on representative inputs.")
    if CouplingKind.TOOL in kinds:
        tests.append("Verify every tool call and tool-result round trip.")
    if kinds & {CouplingKind.STRUCTURED_OUTPUT, CouplingKind.RESPONSE_PARSER}:
        tests.append("Verify structured outputs against downstream parsers and schemas.")
    if CouplingKind.STREAMING in kinds:
        tests.append("Exercise streaming completion and interruption behavior.")
    if CouplingKind.MULTIMODAL in kinds:
        tests.append("Exercise every detected image or document input path.")
    if CouplingKind.RETRY_ERROR_HANDLING in kinds:
        tests.append("Verify target-specific retry and error-handling behavior.")
    tests.append("Run a representative source-versus-target evaluation before production rollout.")
    return tests


def generate_application_migration_plan(
    application: ApplicationAnalysis,
    source: ResolvedModel,
    target: ResolvedModel,
    comparison: ModelComparison,
    invocation: InvocationMigrationSpec,
) -> MigrationPlan:
    """Compose scan, registry, preparation, and validation into one manifest."""
    source_endpoint = _endpoint(source)
    target_endpoint = _endpoint(target)
    prompt_inputs, unknowns = _prompt_inputs(application)
    prompt_changes: list[PromptMigrationSpec] = []
    validation_results = _schema_validation(application)
    validation_results.extend(
        ValidationIssue(
            code="invocation_migration_blocker",
            level=ValidationLevel.BLOCKER,
            message=message,
        )
        for message in invocation.blockers
    )
    validation_results.extend(
        ValidationIssue(
            code="invocation_migration_warning",
            level=ValidationLevel.WARNING,
            message=message,
        )
        for message in invocation.warnings
    )
    if application.prompt_discovery.coverage is not PromptDiscoveryCoverage.RESOLVED:
        validation_results.append(
            ValidationIssue(
                code="incomplete_prompt_coverage",
                level=ValidationLevel.WARNING,
                message=(
                    "Prompt adaptation coverage is incomplete: "
                    f"{application.prompt_discovery.dynamic_consumers} prompt consumer(s) "
                    "have no resolved static prompt source."
                ),
            )
        )
    for prompt_source, component in prompt_inputs:
        prompt_changes.append(
            prepare_prompt_migration(
                source,
                target,
                component.content,
                source_path=prompt_source.path,
                source_component=component.key,
                source_role=component.role,
            )
        )
        validation_results.extend(
            validate_prompt(
                target,
                component.content,
                source_path=prompt_source.path,
                target_platform=target_endpoint.platform,
            ).issues
        )

    target_context = target.effective_capabilities.context_window_tokens
    required_context = application.requirements.minimum_context_window
    if required_context is not None and (
        target_context is None or target_context < required_context
    ):
        validation_results.append(
            ValidationIssue(
                code="context_window_regression",
                level=ValidationLevel.BLOCKER,
                message=(
                    f"Application requires {required_context} context tokens; target provides "
                    f"{target_context if target_context is not None else 'unknown'}."
                ),
            )
        )

    blockers = [*invocation.blockers]
    warnings = [*application.warnings, *invocation.warnings]
    for issue in validation_results:
        if issue.level is ValidationLevel.BLOCKER:
            blockers.append(issue.message)
        elif issue.level is ValidationLevel.WARNING:
            warnings.append(issue.message)
    for assessment in (
        invocation.source_analysis.tool_compatibility,
        invocation.source_analysis.structured_output_compatibility,
    ):
        if assessment and assessment.state is CompatibilityState.UNKNOWN:
            unknowns.append(assessment.rationale)
    for difference in comparison.differences:
        if difference.state.value == "unknown":
            unknowns.append(difference.migration_impact)

    required_changes = [
        _change("invocation", item, _locations(application, CouplingKind.INVOCATION))
        for item in invocation.required_changes
    ]
    optional_changes: list[PlannedMigrationChange] = []
    for prompt in prompt_changes:
        if prompt.candidate_prompt != "":
            suffix = f" ({prompt.source_component})" if prompt.source_component else ""
            optional_changes.append(
                _change(
                    "prompt",
                    f"Review the prepared prompt candidate for {prompt.source_path}{suffix}.",
                    _locations(application, CouplingKind.PROMPT),
                )
            )
    tool_changes = (
        [
            _change(
                "tool",
                invocation.source_analysis.tool_compatibility.rationale,
                _locations(application, CouplingKind.TOOL),
            )
        ]
        if invocation.source_analysis.tool_compatibility
        else []
    )
    output_changes = (
        [
            _change(
                "output_contract",
                invocation.source_analysis.structured_output_compatibility.rationale,
                _locations(
                    application,
                    CouplingKind.STRUCTURED_OUTPUT,
                    CouplingKind.RESPONSE_PARSER,
                ),
            )
        ]
        if invocation.source_analysis.structured_output_compatibility
        else []
    )
    configuration_changes = []
    if (
        source_endpoint.provider != target_endpoint.provider
        or source_endpoint.platform != target_endpoint.platform
    ):
        configuration_changes.append(
            _change(
                "configuration",
                f"Replace {source_endpoint.platform} credentials/configuration with reviewed "
                f"{target_endpoint.platform} configuration.",
                _locations(
                    application,
                    CouplingKind.CONFIGURATION,
                    CouplingKind.RETRY_ERROR_HANDLING,
                ),
            )
        )

    affected_files = sorted(
        {
            *(item.location.path for item in application.findings),
            *(item.source_path for item in prompt_changes if item.source_path),
        }
    )
    complexity: Literal["low", "medium", "high", "blocked"] = (
        "blocked"
        if blockers
        else "high"
        if comparison.highest_severity
        in {
            ComparisonSeverity.HIGH,
            ComparisonSeverity.BREAKING,
        }
        else "medium"
        if required_changes or warnings
        else "low"
    )
    providers = sorted(application.requirements.source_providers)
    platforms = sorted(application.requirements.source_platforms)
    return MigrationPlan(
        source=source_endpoint,
        target=target_endpoint,
        prompt_discovery=application.prompt_discovery,
        application=MigrationApplicationSummary(
            root=application.root,
            files_scanned=application.files_scanned,
            finding_count=len(application.findings),
            providers=providers,
            platforms=platforms,
            invocation_count=sum(
                item.kind is CouplingKind.INVOCATION for item in application.findings
            ),
            prompt_count=sum(item.kind is CouplingKind.PROMPT for item in application.findings),
        ),
        target_selection_rationale=[
            f"The caller selected {target_endpoint.model} on {target_endpoint.platform}.",
            (
                f"Static analysis found {len(blockers)} blocker(s); the target remains a "
                "proposal until they are resolved."
                if blockers
                else "Static analysis found no proven blocker for the detected application "
                "requirements."
            ),
            f"Review {len(unknowns)} unresolved compatibility item(s) before deployment.",
        ],
        model_differences=comparison,
        affected_files=affected_files,
        required_changes=required_changes,
        optional_changes=optional_changes,
        blockers=sorted(set(blockers)),
        warnings=sorted(set(warnings)),
        unknowns=sorted(set(unknowns)),
        prompt_changes=prompt_changes,
        invocation_changes=[invocation],
        tool_changes=tool_changes,
        output_contract_changes=output_changes,
        configuration_changes=configuration_changes,
        validation_results=validation_results,
        required_tests=_required_tests(application),
        rollout_recommendations=[
            "Resolve every blocker and review every unknown before changing traffic.",
            "Apply changes as a reviewable patch; do not overwrite the application from this plan.",
            "Deploy behind a reversible configuration switch and monitor behavior, "
            "errors, latency, and cost.",
            "Retain the source configuration until target validation and rollback checks pass.",
        ],
        migration_complexity=complexity,
        overall_migration_risk=comparison.highest_severity,
    )


def migration_manifest_as_yaml(plan: MigrationPlan) -> str:
    """Render the stable `migration-manifest.yaml` contract."""
    return yaml.safe_dump(
        {"migration": plan.model_dump(mode="json")},
        sort_keys=False,
        allow_unicode=True,
    )


def _candidate_presence(candidate: Any | None) -> str:
    return "candidate emitted" if candidate is not None else "no safe candidate"


def generate_migration_report(plan: MigrationPlan) -> str:
    """Render a concise human review report from the canonical manifest."""
    lines = [
        f"# Migration report: {plan.source.model} → {plan.target.model}",
        "",
        "## Summary",
        "",
        f"- Application: `{plan.application.root}` "
        f"({plan.application.files_scanned} files scanned)",
        f"- Route: `{plan.source.provider}/{plan.source.platform}` → "
        f"`{plan.target.provider}/{plan.target.platform}`",
        f"- Complexity: **{plan.migration_complexity}**",
        f"- Behavioral risk: **{plan.overall_migration_risk.value}**",
        f"- Affected files: {len(plan.affected_files)}",
        f"- Blockers: {len(plan.blockers)}; warnings: {len(plan.warnings)}; "
        f"unknowns: {len(plan.unknowns)}",
        "",
        "## Why this target",
        "",
        *[f"- {item}" for item in plan.target_selection_rationale],
        "",
        "## Important model differences",
        "",
    ]
    material = [
        item
        for item in plan.model_differences.differences
        if item.severity is not ComparisonSeverity.INFO
    ]
    lines.extend(
        [
            f"- **{item.category}** ({item.severity.value}): {item.migration_impact}"
            for item in material
        ]
        or ["- No material model differences were recorded."]
    )
    discovery = plan.prompt_discovery
    discovery_lines = [
        f"Prompt consumers detected: {discovery.consumers}",
        f"Consumers backed by resolved prompt sources: {discovery.source_backed_consumers}",
        f"Inline prompts (adapted through their file tasks): {discovery.inline_consumers}",
        f"Dynamic prompts without a static source: {discovery.dynamic_consumers}",
        f"Resolved prompt sources: {discovery.resolved_sources}; "
        f"low-confidence candidates: {discovery.low_confidence_sources}",
        f"Coverage: **{discovery.coverage.value}**",
    ]
    if discovery.coverage is not PromptDiscoveryCoverage.RESOLVED:
        discovery_lines.append(
            "Warning: prompt adaptation coverage is incomplete; review every unresolved "
            "prompt consumer manually."
        )
    prompt_change_lines = [
        f"Review `{item.source_path or '<inline>'}`"
        + (f" (`{item.source_component}`)" if item.source_component else "")
        + f": {len(item.semantic_diff)} semantic-diff items and "
        f"{len(item.migration_risks)} recorded risks."
        for item in plan.prompt_changes
    ]
    if not prompt_change_lines and discovery.consumers:
        prompt_change_lines = [
            "Prompt adaptation coverage is incomplete: no static prompt source was resolved "
            f"for the {discovery.consumers} detected prompt consumer(s)."
            if discovery.coverage is not PromptDiscoveryCoverage.RESOLVED
            else "No prompt files require preparation; detected prompt content is inline "
            "and is adapted through its file tasks."
        ]
    sections = (
        ("Prompt discovery", discovery_lines),
        ("Affected files", [f"`{item}`" for item in plan.affected_files]),
        ("Blockers", plan.blockers),
        ("Warnings", plan.warnings),
        ("Unresolved unknowns", plan.unknowns),
        ("Required changes", [item.description for item in plan.required_changes]),
        ("Optional changes", [item.description for item in plan.optional_changes]),
        ("Prompt changes", prompt_change_lines),
        ("Tool changes", [item.description for item in plan.tool_changes]),
        (
            "Tool schema candidates",
            [
                f"`{item.name or '<unknown>'}` → `{item.target_field or '<unsupported>'}` "
                f"({item.state.value}); "
                f"{_candidate_presence(item.target_definition)}"
                for invocation in plan.invocation_changes
                for item in invocation.tool_schema_migrations
            ],
        ),
        (
            "Output-contract changes",
            [item.description for item in plan.output_contract_changes],
        ),
        (
            "Structured-output candidates",
            [
                f"`{item.name or '<unnamed>'}` → "
                f"`{item.target_field or '<unsupported>'}` ({item.state.value}); "
                f"{_candidate_presence(item.target_configuration)}"
                for invocation in plan.invocation_changes
                for item in invocation.structured_output_migrations
            ],
        ),
        (
            "Configuration changes",
            [item.description for item in plan.configuration_changes],
        ),
        (
            "Target platform configuration",
            [
                f"`{item.target_platform}` via `{item.target_sdk}.{item.target_client}` "
                f"using `{item.credential_strategy}` ({item.state.value})."
                for item in (
                    invocation.configuration_migration
                    for invocation in plan.invocation_changes
                    if invocation.configuration_migration is not None
                )
            ],
        ),
        (
            "Validation results",
            [
                f"[{item.level.value}] {item.code}: {item.message}"
                for item in plan.validation_results
            ],
        ),
        ("What to test before deployment", plan.required_tests),
        ("Rollout recommendations", plan.rollout_recommendations),
    )
    for title, items in sections:
        lines.extend(("", f"## {title}", ""))
        lines.extend([f"- {item}" for item in items] or ["- None."])
    lines.extend(
        (
            "",
            "This report and its manifest are review artifacts. "
            "No analyzed source files were modified during plan generation.",
            "",
        )
    )
    return "\n".join(lines)
