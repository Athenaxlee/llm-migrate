"""Application-level V0.4 migration planning and reporting."""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from llm_migrate.core.comparison import capability_evidence_url
from llm_migrate.core.migration import prepare_prompt_migration, validate_prompt
from llm_migrate.core.models import (
    ApplicationAnalysis,
    BlockerCategory,
    ComparisonSeverity,
    CompatibilityState,
    CouplingKind,
    InvocationMigrationSpec,
    MigrationApplicationSummary,
    MigrationBlocker,
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
    ResolutionKind,
    ResolvedModel,
    SourceLocation,
    ValidationIssue,
    ValidationLevel,
    dedupe_blockers,
    migration_blocker,
    migration_complexity,
    prompt_issue_blocker,
)
from llm_migrate.core.resolver import normalize_identifier, strip_identifier_decorations


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


# Capability differences that matter only when the application uses the
# capability: with no detected coupling of the governing kind, an unknown
# about them is already answered by the scan, and a breaking difference is a
# report fact rather than a finding.
_USAGE_GATED_CATEGORIES = frozenset(
    {"tool_use", "parallel_tool_use", "structured_output", "streaming", "multimodality"}
)


def difference_unused_by_application(difference: ModelDifference) -> bool:
    """Whether the scan shows the application does not use this capability."""
    return difference.category in _USAGE_GATED_CATEGORIES and not difference.locations


def difference_governs_prompts(difference: ModelDifference) -> bool:
    """Whether a model difference's governing surface is prompt content.

    Prompt-governing differences travel as prompt guidance; every other
    material difference attaches a required change to the files whose
    detected couplings it governs (the one propagation mechanism).
    """
    if difference.category == "behavioral_prompt_guidance":
        return True
    if difference.category == "migration_knowledge":
        return str(difference.source_value or "").startswith("prompt_guidance")
    return False


_DIFFERENCE_CHANGE_CATEGORIES: dict[str, str] = {
    "parameters": "parameter",
    "reasoning": "parameter",
    "context_output_limits": "parameter",
    "tool_use": "tool",
    "parallel_tool_use": "tool",
    "structured_output": "output_contract",
    "multimodality": "multimodal",
    "streaming": "invocation",
    "prompt_caching": "invocation",
    "batch_inference": "invocation",
}


def _difference_change_category(difference: ModelDifference) -> str:
    category = difference.category
    if category == "migration_knowledge":
        field_path = str(difference.source_value or "")
        if field_path.startswith("parameters."):
            return "parameter"
        if field_path.startswith("capabilities."):
            capability = field_path.split(".", 1)[1]
            return _DIFFERENCE_CHANGE_CATEGORIES.get(capability, "configuration")
        return "invocation"
    return _DIFFERENCE_CHANGE_CATEGORIES.get(category, "configuration")


def _difference_text(difference: ModelDifference) -> str:
    return (
        f"Model difference ({difference.severity.value}): {difference.migration_impact}"
        + (f" Action: {difference.recommended_action}" if difference.recommended_action else "")
        + (
            f" (evidence: {difference.target_evidence_url})"
            if difference.target_evidence_url
            else ""
        )
    )


def _difference_changes(comparison: ModelComparison) -> list[PlannedMigrationChange]:
    """One required change per material difference, on the files it governs.

    The single difference-propagation mechanism: a non-prompt difference with
    mapped locations becomes a required change on exactly those files;
    prompt-governing differences travel through prompt guidance instead, and a
    difference with no mapped coupling stays a report-level fact (nothing to
    attach it to — never a phantom task).
    """
    return [
        _change(
            _difference_change_category(difference),
            _difference_text(difference),
            difference.locations,
        )
        for difference in comparison.differences
        if difference.severity is not ComparisonSeverity.INFO
        and difference.locations
        and not difference_governs_prompts(difference)
    ]


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


def out_of_scope_paths(plan: MigrationPlan) -> set[str]:
    """Application paths the plan classified out of scope."""
    return {item.split(": ", 1)[0] for item in plan.out_of_scope}


def _model_spellings(model: ResolvedModel) -> set[str]:
    """Every normalized reviewed spelling of a model across its platforms."""
    identity = model.profile.identity
    names = {identity.canonical_name, *identity.aliases}
    for platform in model.profile.platforms:
        names.add(platform.model_id)
        if platform.invocation is not None:
            names.update(selector.model_id for selector in platform.invocation.selectors)
    return {normalize_identifier(strip_identifier_decorations(item)) for item in names if item}


def _prompt_inputs(
    application: ApplicationAnalysis,
    source: ResolvedModel,
    target: ResolvedModel,
) -> tuple[list[tuple[PromptSource, PromptComponent]], list[str], list[str]]:
    """Prompt components to prepare, unknowns, and out-of-scope prompt files.

    Two deterministic scoping rules keep files this migration does not govern
    out of the unknowns: a config-referenced prompt whose referencing
    profiles all declare some OTHER model is out of scope, and an
    unreferenced candidate is out of scope when a sibling prompt file in the
    same directory was selected by code, configuration, or override (the
    application demonstrably selects among them). An unreferenced candidate
    with no selected sibling stays an unknown — nothing shows whether it is
    used.
    """
    inputs: list[tuple[PromptSource, PromptComponent]] = []
    unknowns: list[str] = []
    out_of_scope: list[str] = []
    governed = _model_spellings(source) | _model_spellings(target)
    prepared: list[PromptSource] = []
    for item in application.prompt_sources:
        if item.confidence is PromptSourceConfidence.LOW:
            continue
        foreign = [
            model_id
            for model_id in item.profile_model_ids
            if normalize_identifier(strip_identifier_decorations(model_id)) not in governed
        ]
        if item.profile_model_ids and len(foreign) == len(item.profile_model_ids):
            out_of_scope.append(
                f"{item.path}: referenced only by configuration profile(s) for "
                f"{', '.join(foreign)}, which this migration does not cover; not prepared."
            )
            continue
        prepared.append(item)
        inputs.extend((item, component) for component in item.components)
    selected_directories = {posixpath.dirname(item.path) for item in prepared}
    for item in application.prompt_sources:
        if item.confidence is not PromptSourceConfidence.LOW:
            continue
        directory = posixpath.dirname(item.path)
        if directory in selected_directories:
            siblings = sorted(
                other.path for other in prepared if posixpath.dirname(other.path) == directory
            )
            out_of_scope.append(
                f"{item.path}: not referenced by the application's code or configuration, "
                f"while sibling prompt file(s) {', '.join(siblings)} are; treated as out of "
                "scope. Include it with add_prompt_sources (or prompt_sources at start) if "
                "it is live."
            )
            continue
        unknowns.append(
            f"{item.path} contains prompt-like keys but nothing references it; it was "
            "not prepared automatically. Include it with add_prompt_sources (or "
            "prompt_sources at start) if it is live."
        )
    discovery = application.prompt_discovery
    if discovery.dynamic_consumers:
        unknowns.append(
            f"{discovery.dynamic_consumers} prompt consumer(s) supply dynamically built "
            "content with no statically resolvable prompt source; review them manually."
        )
    return inputs, unknowns, out_of_scope


def _schema_validation(application: ApplicationAnalysis) -> list[ValidationIssue]:
    """Schema validity issues for the report; the invocation analyzer owns the
    matching blockers (one per contract, located and discriminated), so the
    same invalid schema never becomes two differently-worded blockers."""
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


def _prompt_issue_blocker(
    issue: ValidationIssue,
    spec: PromptMigrationSpec,
    target: ResolvedModel,
    target_platform: str,
) -> MigrationBlocker:
    """Structure one blocker-level prompt validation issue.

    The id derivation is shared with the submission gate through
    `prompt_issue_blocker`; the location ties the blocker (and any redesign
    task a decision injects) to the prompt file so it reaches the worklist.
    """
    data: dict[str, Any] = {"source_path": issue.source_path}
    evidence: list[str] = []
    if issue.code == "context_window_exceeded":
        data["approximate_tokens"] = spec.source_prompt_analysis.approximate_token_count
        data["target_context_window"] = target.effective_capabilities.context_window_tokens
        url = capability_evidence_url(target.profile, target.platform, "context_window_tokens")
        if url:
            evidence = [url]
    if issue.code == "invalid_target_platform":
        data["platform"] = target_platform
    locations = (
        [SourceLocation(path=issue.source_path, line=1, column=0)] if issue.source_path else []
    )
    return prompt_issue_blocker(issue, evidence_urls=evidence, locations=locations, data=data)


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
    prompt_inputs, unknowns, out_of_scope = _prompt_inputs(application, source, target)
    prompt_changes: list[PromptMigrationSpec] = []
    validation_results = _schema_validation(application)
    blockers: list[MigrationBlocker] = [*invocation.blockers]
    validation_results.extend(
        ValidationIssue(
            code="invocation_migration_blocker",
            level=ValidationLevel.BLOCKER,
            message=blocker.message,
        )
        for blocker in invocation.blockers
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
        spec = prepare_prompt_migration(
            source,
            target,
            component.content,
            source_path=prompt_source.path,
            source_component=component.key,
            source_role=component.role,
        )
        prompt_changes.append(spec)
        prompt_validation = validate_prompt(
            target,
            component.content,
            source_path=prompt_source.path,
            target_platform=target_endpoint.platform,
        )
        validation_results.extend(prompt_validation.issues)
        blockers.extend(
            _prompt_issue_blocker(issue, spec, target, target_endpoint.platform)
            for issue in prompt_validation.issues
            if issue.level is ValidationLevel.BLOCKER
        )

    target_context = target.effective_capabilities.context_window_tokens
    required_context = application.requirements.minimum_context_window
    if required_context is not None and (
        target_context is None or target_context < required_context
    ):
        regression_message = (
            f"Application requires {required_context} context tokens; target provides "
            f"{target_context if target_context is not None else 'unknown'}."
        )
        validation_results.append(
            ValidationIssue(
                code="context_window_regression",
                level=ValidationLevel.BLOCKER,
                message=regression_message,
            )
        )
        context_url = capability_evidence_url(
            target.profile, target.platform, "context_window_tokens"
        )
        blockers.append(
            migration_blocker(
                code="context_window_regression",
                category=BlockerCategory.CONTEXT_WINDOW,
                message=regression_message,
                evidence_urls=[context_url] if context_url else [],
                locations=_locations(application, CouplingKind.PROMPT),
                data={
                    "required_context_window": required_context,
                    "target_context_window": target_context,
                },
            )
        )

    blockers = dedupe_blockers(blockers)
    warnings = [*application.warnings, *invocation.warnings]
    warnings.extend(
        issue.message for issue in validation_results if issue.level is ValidationLevel.WARNING
    )
    for assessment in (
        invocation.source_analysis.tool_compatibility,
        invocation.source_analysis.structured_output_compatibility,
    ):
        if assessment and assessment.state is CompatibilityState.UNKNOWN:
            unknowns.append(assessment.rationale)
    for difference in comparison.differences:
        if difference.state.value == "unknown" and not difference_unused_by_application(difference):
            unknowns.append(difference.migration_impact)

    required_changes = [
        *(
            _change("invocation", item, _locations(application, CouplingKind.INVOCATION))
            for item in invocation.required_changes
        ),
        *_difference_changes(comparison),
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
    # Files whose only detected coupling is an incidental SDK import carry no
    # migration work of their own; the worklist lists them as unaffected
    # instead of manufacturing a generic adaptation task per file.
    findings_by_file: dict[str, set[CouplingKind]] = {}
    for finding in application.findings:
        findings_by_file.setdefault(finding.location.path, set()).add(finding.kind)
    incidental_files = sorted(
        file for file, kinds in findings_by_file.items() if kinds <= {CouplingKind.PROVIDER_SDK}
    )
    complexity = migration_complexity(
        blockers, comparison.highest_severity, required_changes, warnings
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
        incidental_files=incidental_files,
        required_changes=required_changes,
        optional_changes=optional_changes,
        blockers=blockers,
        warnings=sorted(set(warnings)),
        unknowns=sorted(set(unknowns)),
        out_of_scope=sorted(set(out_of_scope)),
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
    if not links and difference_unused_by_application(difference):
        return "— (not used by this application)"
    return "<br>".join(links) if links else "—"


_DECISION_KIND_LABELS = {
    ResolutionKind.RETARGET: "Retarget",
    ResolutionKind.REDESIGN_TASK: "Redesign",
    ResolutionKind.CORRECTION: "Correction",
    ResolutionKind.ACCEPT_WITH_RATIONALE: "ACCEPTED RISK",
}


def _decision_lines(plan: MigrationPlan) -> list[str]:
    """Render every recorded blocker decision, keeping stale ones loud."""
    lines: list[str] = []
    for applied in plan.decisions:
        decision = applied.decision
        label = _DECISION_KIND_LABELS[decision.kind]
        rationale = (
            f" Rationale: {decision.rationale}"
            if decision.rationale and decision.rationale != decision.summary
            else ""
        )
        if applied.status == "stale":
            lines.append(
                f"**STALE decision** ({label}, {decision.decided_on.isoformat()}) for blocker "
                f"`{decision.blocker_code}`: {applied.detail} The decision was NOT applied; "
                "re-run blocker resolution if the blocker still exists."
            )
        elif applied.status == "superseded":
            lines.append(
                f"**Superseded** ({label}, {decision.decided_on.isoformat()}) for blocker "
                f"`{decision.blocker_code}`: {decision.summary}{rationale} {applied.detail}"
            )
        else:
            lines.append(
                f"**{label}** ({decision.decided_on.isoformat()}) resolved blocker "
                f"`{decision.blocker_code}` ({decision.blocker_message}) — "
                f"{decision.summary}{rationale} Result: {applied.detail}"
            )
    return lines


_CANDIDATE_MARKER = " contains prompt-like keys but nothing references it"


def unreferenced_prompt_candidates(plan: MigrationPlan) -> list[str]:
    """Candidate prompt files the plan reports as unreferenced unknowns."""
    return sorted(
        item.split(_CANDIDATE_MARKER, 1)[0] for item in plan.unknowns if _CANDIDATE_MARKER in item
    )


def apply_prompt_discovery_dismissals(
    plan: MigrationPlan,
    dismissed: Mapping[str, str],
    dismissed_consumers: Sequence[str] = (),
) -> MigrationPlan:
    """Move user-dismissed candidates and consumers out of the unknowns.

    A dismissed candidate prompt file (recorded with the user's rationale in
    `migration.yaml`) leaves the unknowns and is listed out of scope with
    that rationale — replacing any heuristic out-of-scope line for the same
    file, since the user's recorded reason is the stronger evidence.
    Dismissed consumers the scan applied are listed the same way. Nothing is
    dropped silently: every dismissal stays visible.
    """
    if not dismissed:
        return plan
    unknowns: list[str] = []
    out_of_scope: list[str] = []

    def dismissal(path: str) -> str:
        return f"{path}: dismissed by the user as not a live prompt — {dismissed[path]}"

    for item in plan.out_of_scope:
        scoped = item.split(": ", 1)[0]
        out_of_scope.append(dismissal(scoped) if scoped in dismissed else item)
    for item in plan.unknowns:
        candidate = item.split(_CANDIDATE_MARKER, 1)[0] if _CANDIDATE_MARKER in item else None
        if candidate is not None and candidate in dismissed:
            out_of_scope.append(dismissal(candidate))
            continue
        unknowns.append(item)
    out_of_scope.extend(
        f"{location}: prompt consumer dismissed by the user as runtime-built content — "
        f"{dismissed[location]}"
        for location in dismissed_consumers
        if location in dismissed
    )
    return plan.model_copy(update={"unknowns": unknowns, "out_of_scope": sorted(set(out_of_scope))})


def unknown_action(unknown: str) -> str:
    """The concrete next action for one unresolved unknown."""
    if _CANDIDATE_MARKER in unknown:
        path = unknown.split(_CANDIDATE_MARKER, 1)[0]
        return (
            f"If `{path}` is a live prompt, include it on the live run with "
            f'add_prompt_sources(run_dir, ["{path}"]); otherwise dismiss it with '
            f'add_prompt_sources(run_dir, [], dismiss=["{path}"], rationale="...").'
        )
    if "prompt consumer(s) supply dynamically built content" in unknown:
        return (
            "Review each dynamic consumer (get_run_status lists them): "
            "confirm_prompt_consumer(run_dir, location, source_path) when it reads a "
            "static prompt file, or dismiss it with a rationale when the content is "
            "genuinely runtime-built."
        )
    if unknown.startswith("One or both values for "):
        field = unknown.split("'")[1] if "'" in unknown else "this capability"
        return (
            f"Verify `{field}` on the target before deployment (a BYOK evaluation, or "
            "bounded research for the missing registry fact)."
        )
    return "Review manually before deployment."


def _unknowns_table(unknowns: list[str]) -> list[str]:
    lines = ["", "## Unresolved unknowns", ""]
    if not unknowns:
        return [*lines, "- None."]
    lines.extend(("| Unknown | Action |", "|---|---|"))
    lines.extend(
        f"| {_table_text(item, limit=240)} | {_table_text(unknown_action(item), limit=240)} |"
        for item in unknowns
    )
    return lines


def generate_migration_report(plan: MigrationPlan) -> str:
    """Render a concise human review report from the canonical manifest."""
    applied_decisions = sum(item.status == "applied" for item in plan.decisions)
    accepted_decisions = sum(
        item.status == "applied" and item.decision.kind is ResolutionKind.ACCEPT_WITH_RATIONALE
        for item in plan.decisions
    )
    superseded_decisions = sum(item.status == "superseded" for item in plan.decisions)
    stale_decisions = sum(item.status == "stale" for item in plan.decisions)
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
        f"- Unresolved blockers: {len(plan.blockers)}; warnings: {len(plan.warnings)}; "
        f"unknowns: {len(plan.unknowns)}",
        *(
            [
                f"- Blocker decisions: {applied_decisions} applied "
                f"({accepted_decisions} accepted risk), "
                f"{superseded_decisions} superseded, {stale_decisions} stale"
            ]
            if plan.decisions
            else []
        ),
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
    discovery_lines.extend(f"Out of scope: {item}" for item in plan.out_of_scope)
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
        ("Blockers", [item.rendered for item in plan.blockers]),
        ("Decisions", _decision_lines(plan)),
        ("Warnings", plan.warnings),
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
        if title == "Warnings":
            lines.extend(_unknowns_table(plan.unknowns))
    lines.extend(
        (
            "",
            "This report and its manifest are review artifacts. "
            "No analyzed source files were modified during plan generation.",
            "",
        )
    )
    return "\n".join(lines)
