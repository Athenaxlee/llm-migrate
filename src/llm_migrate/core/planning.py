"""Application-level V0.4 migration planning and reporting."""

from __future__ import annotations

import json
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
    ModelDifference,
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


def _parameter_locations(application: ApplicationAnalysis, names: set[str]) -> list[SourceLocation]:
    return sorted(
        {
            item.location
            for item in application.findings
            if item.kind is CouplingKind.PARAMETER and (item.metadata or {}).get("name") in names
        },
        key=lambda item: (item.path, item.line, item.column),
    )


_CATEGORY_LOCATION_KINDS: dict[str, tuple[CouplingKind, ...]] = {
    "identity": (CouplingKind.PROVIDER_SDK, CouplingKind.CONFIGURATION),
    "platform": (CouplingKind.MODEL_IDENTIFIER, CouplingKind.CONFIGURATION),
    "lifecycle": (CouplingKind.MODEL_IDENTIFIER,),
    "tool_use": (CouplingKind.TOOL,),
    "parallel_tool_use": (CouplingKind.TOOL,),
    "structured_output": (CouplingKind.STRUCTURED_OUTPUT, CouplingKind.RESPONSE_PARSER),
    "streaming": (CouplingKind.STREAMING,),
    "behavioral_prompt_guidance": (CouplingKind.PROMPT,),
    "text_input": (CouplingKind.INVOCATION,),
    "prompt_caching": (CouplingKind.INVOCATION,),
    "batch_inference": (CouplingKind.INVOCATION,),
}
_REASONING_PARAMETERS = {"thinking", "reasoning_effort", "effort"}
_OUTPUT_LIMIT_PARAMETERS = {"max_tokens", "max_output_tokens"}


def _difference_locations(
    application: ApplicationAnalysis, difference: ModelDifference
) -> list[SourceLocation]:
    """Best-effort mapping from a model difference to the code it affects."""
    category, field = difference.category, difference.field or ""
    if category == "migration_knowledge":
        field_path = str(difference.source_value or "")
        if field_path.startswith("parameters."):
            category, field = "parameters", field_path.split(".", 1)[1]
        elif field_path.startswith("capabilities."):
            category, field = field_path.split(".", 1)[1], ""
        elif field_path.startswith("prompt_guidance"):
            return _locations(application, CouplingKind.PROMPT)
        else:
            return _locations(application, CouplingKind.INVOCATION)
    if category == "parameters":
        return _parameter_locations(application, {field})
    if category == "reasoning":
        return _parameter_locations(application, _REASONING_PARAMETERS)
    if category == "multimodality":
        return sorted(
            {
                item.location
                for item in application.findings
                if item.kind is CouplingKind.MULTIMODAL and item.value == field
            },
            key=lambda item: (item.path, item.line, item.column),
        )
    if category == "context_output_limits":
        if field == "maximum_output_tokens":
            return _parameter_locations(application, _OUTPUT_LIMIT_PARAMETERS)
        return _locations(application, CouplingKind.PROMPT)
    kinds = _CATEGORY_LOCATION_KINDS.get(category)
    return _locations(application, *kinds) if kinds else []


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
    comparison = comparison.model_copy(
        update={
            "differences": [
                (
                    difference.model_copy(update={"locations": locations})
                    if (locations := _difference_locations(application, difference))
                    else difference
                )
                for difference in comparison.differences
            ]
        }
    )
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
                    f"Adapt and review the prompt {prompt.source_path}{suffix} for the "
                    "target model using its evidence-linked guidance.",
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


def _table_text(text: str, limit: int = 100) -> str:
    """Make text safe inside one Markdown table cell, including link text."""
    flattened = " ".join(text.split())
    if len(flattened) > limit:
        flattened = flattened[: limit - 1] + "…"
    return flattened.replace("|", "\\|").replace("[", "\\[").replace("]", "\\]")


def _difference_type(difference: ModelDifference) -> str:
    category = difference.category.replace("_", " ")
    field = (difference.field or "").replace("_", " ")
    if field and field != category:
        return f"{category} ({field})"
    return category


def _difference_value(value: Any) -> str:
    """Compact, table-safe rendering of one side of a model difference."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _table_text(value)
    if isinstance(value, dict):
        if {"platform", "model_id"} <= value.keys():
            endpoint = f" ({value['endpoint']})" if value.get("endpoint") else ""
            return _table_text(f"`{value['model_id']}` on {value['platform']}{endpoint}")
        if value and set(value) <= {"input", "output", "cached_input", "batch"}:
            parts = [
                f"{name.replace('_', ' ')} ${component['amount']}/M tokens"
                for name, component in value.items()
                if isinstance(component, dict) and component.get("amount") is not None
            ]
            return _table_text(", ".join(parts)) if parts else "unknown"
        if value and all(isinstance(item, list) for item in value.values()):
            count = sum(len(item) for item in value.values())
            return f"{count} guidance item(s)"
    return _table_text(json.dumps(value, default=str, sort_keys=True))


def _difference_claim(value: Any, evidence_url: str | None) -> str:
    """A claim cell, hyperlinked to the registry source that backs it.

    Absent or unknown values are never linked: a source can back a claim,
    not the lack of one.
    """
    cell = _difference_value(value)
    if evidence_url is not None and value is not None and cell not in ("", "unknown"):
        return f"[{cell}]({evidence_url})"
    return cell


def _difference_files(difference: ModelDifference, limit: int = 3) -> str:
    """Hyperlinked file:line locations, relative to the application root."""
    unique = sorted({(item.path, item.line) for item in difference.locations})
    links = [f"[{path}:{line}]({path}#L{line})" for path, line in unique[:limit]]
    if len(unique) > limit:
        links.append(f"+{len(unique) - limit} more")
    return "<br>".join(links) if links else "—"


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
    severity_rank = {severity: index for index, severity in enumerate(ComparisonSeverity)}
    material = sorted(
        (
            item
            for item in plan.model_differences.differences
            if item.severity is not ComparisonSeverity.INFO
        ),
        key=lambda item: (-severity_rank[item.severity], item.category, item.field or ""),
    )
    if material:
        lines.extend(
            (
                f"| Type | Priority | {plan.source.model} (source) | "
                f"{plan.target.model} (target) | Recommended action | Changed files |",
                "|---|---|---|---|---|---|",
            )
        )
        lines.extend(
            f"| {_table_text(_difference_type(item))} "
            f"| {item.severity.value} "
            f"| {_difference_claim(item.source_value, item.source_evidence_url)} "
            f"| {_difference_claim(item.target_value, item.target_evidence_url)} "
            f"| {_table_text(item.recommended_action or item.migration_impact, limit=160)} "
            f"| {_difference_files(item)} |"
            for item in material
        )
    else:
        lines.append("- No material model differences were recorded.")
    informational = len(plan.model_differences.differences) - len(material)
    if informational:
        lines.extend(
            (
                "",
                f"{informational} informational difference(s) (unchanged or "
                "migration-neutral facts) are recorded in the manifest.",
            )
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
