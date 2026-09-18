"""Deterministic invocation and adjacent contract migration analysis."""

from __future__ import annotations

from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from llm_migrate.core.comparison import capability_evidence_url, evidence_url
from llm_migrate.core.models import (
    ApplicationAnalysis,
    BlockerCategory,
    CompatibilityAssessment,
    CompatibilityState,
    CouplingKind,
    InvocationAnalysis,
    InvocationMigrationSpec,
    MigrationBlocker,
    ParameterMapping,
    ParameterState,
    ParserAssumption,
    PlatformConfigurationMigration,
    ResolvedModel,
    SourceLocation,
    StructuredOutputContract,
    StructuredOutputMigration,
    TargetInvocation,
    ToolDefinition,
    ToolSchemaMigration,
    dedupe_blockers,
    migration_blocker,
)


def _values(application: ApplicationAnalysis, kind: CouplingKind) -> list[str]:
    return sorted(
        {
            str(item.value)
            for item in application.findings
            if item.kind is kind and item.value is not None
        }
    )


def _schema_issues(schema: dict[str, Any] | None) -> list[str]:
    if schema is None:
        return ["Schema could not be resolved statically."]
    issues: list[str] = []
    schema_type = schema.get("type")
    if schema_type is not None and not isinstance(schema_type, (str, list)):
        issues.append("JSON Schema type must be a string or list.")
    if (
        schema_type == "object"
        and "properties" in schema
        and not isinstance(schema["properties"], dict)
    ):
        issues.append("Object-schema properties must be a mapping.")
    advanced_keywords = {"$ref", "$dynamicRef", "oneOf", "anyOf", "allOf", "not"}
    advanced: set[str] = set()

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            advanced.update(str(key) for key in value if key in advanced_keywords)
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)

    inspect(schema)
    if advanced:
        issues.append(
            "Advanced JSON Schema keywords require semantic review: " + ", ".join(sorted(advanced))
        )
    return issues


def _schema_validation_error(schema: dict[str, Any] | None) -> str | None:
    if schema is None:
        return None
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return f"Invalid Draft 2020-12 JSON Schema: {exc.message}"
    return None


def _dynamic_invocation_assessment(
    application: ApplicationAnalysis, concern: str
) -> CompatibilityAssessment | None:
    """UNKNOWN assessment when a request is built dynamically (`**kwargs`).

    A splatted or non-literal request hides its fields from static analysis;
    the absence of tool/structured-output findings is then no evidence of
    absence, so the compatibility question stays explicitly unresolved
    instead of silently passing.
    """
    dynamic = [
        item
        for item in application.findings
        if item.kind is CouplingKind.INVOCATION and (item.metadata or {}).get("dynamic_request")
    ]
    if not dynamic:
        return None
    return CompatibilityAssessment(
        concern=concern,
        state=CompatibilityState.UNKNOWN,
        rationale=(
            f"An invocation builds its request dynamically (**kwargs); {concern} "
            "could not be statically established and must be verified manually."
        ),
        locations=[item.location for item in dynamic],
    )


def _tool_assessment(
    application: ApplicationAnalysis, target: ResolvedModel
) -> CompatibilityAssessment | None:
    findings = [item for item in application.findings if item.kind is CouplingKind.TOOL]
    if not findings:
        return _dynamic_invocation_assessment(application, "native tool use")
    capability = target.effective_capabilities.tool_use
    if capability is False:
        return CompatibilityAssessment(
            concern="tool use",
            state=CompatibilityState.UNSUPPORTED,
            rationale="The application uses tools, but the target declares tool use unsupported.",
            locations=[item.location for item in findings],
        )
    if capability is None:
        return CompatibilityAssessment(
            concern="tool use",
            state=CompatibilityState.UNKNOWN,
            rationale="The application uses tools; target tool support is unknown.",
            locations=[item.location for item in findings],
        )
    definitions: list[ToolDefinition] = application.tool_definitions
    issues: list[str] = []
    if not definitions:
        issues.append("Tool definitions could not be resolved statically.")
    for definition in definitions:
        if not definition.name:
            issues.append("A tool definition has no statically resolved name.")
        issues.extend(
            f"Tool {definition.name or '<unknown>'}: {issue}"
            for issue in _schema_issues(definition.input_schema)
        )
        if definition.strict is True:
            issues.append(
                f"Tool {definition.name or '<unknown>'} requires strict-schema semantics."
            )
        if definition.choice not in {None, "auto", "any", "required", "none", "tool"}:
            issues.append(f"Tool-choice mode {definition.choice!r} requires semantic review.")
    target_platform = target.platform.platform if target.platform else None
    same_format = bool(target_platform) and all(
        item.provider_format.startswith(f"{target_platform}.") for item in definitions
    )
    if any("could not be resolved" in item or "no statically resolved" in item for item in issues):
        state = CompatibilityState.UNKNOWN
        rationale = "Tool compatibility cannot be established without complete definitions."
    elif issues:
        state = CompatibilityState.SEMANTICALLY_DIFFERENT
        rationale = "Tool schemas or choice semantics require human review."
    elif same_format:
        state = CompatibilityState.DIRECTLY_COMPATIBLE
        rationale = "Tool definitions already use the selected target-platform format."
    else:
        state = CompatibilityState.MECHANICALLY_CONVERTIBLE
        rationale = "Tool wrappers can be converted while preserving the resolved JSON Schemas."
    return CompatibilityAssessment(
        concern="tool use",
        state=state,
        rationale=rationale,
        locations=[item.location for item in findings],
        issues=sorted(set(issues)),
    )


def _parser_assumptions(application: ApplicationAnalysis) -> list[ParserAssumption]:
    grouped: dict[str, list[SourceLocation]] = {}
    for finding in application.findings:
        if finding.kind is CouplingKind.RESPONSE_PARSER and isinstance(finding.value, str):
            grouped.setdefault(finding.value, []).append(finding.location)
    assumptions: list[ParserAssumption] = []
    for parser, locations in sorted(grouped.items()):
        if "model_validate_json" in parser:
            expectation = "Response must satisfy an application schema."
            strictness: Literal["syntax", "schema", "unknown"] = "schema"
        elif "json.loads" in parser:
            expectation = "Response must be syntactically valid JSON."
            strictness = "syntax"
        else:
            expectation = "Parser behavior requires manual inspection."
            strictness = "unknown"
        assumptions.append(
            ParserAssumption(
                parser=parser,
                expectation=expectation,
                strictness=strictness,
                locations=locations,
            )
        )
    return assumptions


def _structured_output_assessment(
    application: ApplicationAnalysis, target: ResolvedModel
) -> CompatibilityAssessment | None:
    findings = [
        item for item in application.findings if item.kind is CouplingKind.STRUCTURED_OUTPUT
    ]
    if not findings:
        return _dynamic_invocation_assessment(application, "native structured-output use")
    capability = target.effective_capabilities.structured_output
    if capability is False:
        return CompatibilityAssessment(
            concern="structured output",
            state=CompatibilityState.UNSUPPORTED,
            rationale=(
                "The application configures structured output, but the target declares it "
                "unsupported."
            ),
            locations=[item.location for item in findings],
        )
    if capability is None:
        return CompatibilityAssessment(
            concern="structured output",
            state=CompatibilityState.UNKNOWN,
            rationale="Target structured-output support is unknown.",
            locations=[item.location for item in findings],
        )
    contracts: list[StructuredOutputContract] = application.structured_outputs
    issues: list[str] = []
    if not contracts:
        issues.append("Structured-output configuration could not be resolved statically.")
    for contract in contracts:
        issues.extend(_schema_issues(contract.json_schema))
        if contract.strict is True:
            issues.append("The source requires strict structured-output semantics.")
    parsers = _parser_assumptions(application)
    if any(item.strictness == "schema" for item in parsers) and any(
        item.json_schema is None for item in contracts
    ):
        issues.append("A schema-validating parser was found, but its output schema is unresolved.")
    if any(item.strictness == "unknown" for item in parsers):
        issues.append("A response parser has unknown strictness semantics.")
    target_platform = target.platform.platform if target.platform else None
    same_format = bool(target_platform) and all(
        item.provider_format.startswith(f"{target_platform}.") for item in contracts
    )
    if any("could not be resolved" in item or "unresolved" in item for item in issues):
        state = CompatibilityState.UNKNOWN
        rationale = "Structured-output compatibility cannot be established statically."
    elif issues:
        state = CompatibilityState.SEMANTICALLY_DIFFERENT
        rationale = "Schema strictness or parser behavior requires human review."
    elif same_format:
        state = CompatibilityState.DIRECTLY_COMPATIBLE
        rationale = "The output schema already uses the selected target-platform format."
    else:
        state = CompatibilityState.MECHANICALLY_CONVERTIBLE
        rationale = "The resolved JSON Schema can be moved to the target wrapper."
    return CompatibilityAssessment(
        concern="structured output",
        state=state,
        rationale=rationale,
        locations=[item.location for item in findings],
        issues=sorted(set(issues)),
    )


def analyze_invocation(
    application: ApplicationAnalysis,
    target: ResolvedModel | None = None,
) -> InvocationAnalysis:
    parameters: dict[str, str | bool | int | float | None] = {}
    for finding in application.findings:
        if finding.kind is not CouplingKind.PARAMETER:
            continue
        if finding.metadata and isinstance(finding.metadata.get("name"), str):
            name = str(finding.metadata["name"])
            raw = finding.metadata.get("value")
            parameters[name] = raw if isinstance(raw, (str, bool, int, float)) else None
        elif isinstance(finding.value, str):
            name, separator, raw = finding.value.partition("=")
            parameters[name] = raw if separator else None
    all_locations = [
        item.location
        for item in application.findings
        if item.kind
        in {
            CouplingKind.INVOCATION,
            CouplingKind.MODEL_IDENTIFIER,
            CouplingKind.PARAMETER,
            CouplingKind.TOOL,
            CouplingKind.STRUCTURED_OUTPUT,
            CouplingKind.STREAMING,
            CouplingKind.MULTIMODAL,
        }
    ]
    parser_assumptions = _parser_assumptions(application)
    return InvocationAnalysis(
        application_root=application.root,
        providers=sorted(application.requirements.source_providers),
        platforms=sorted(application.requirements.source_platforms),
        sdk_imports=_values(application, CouplingKind.PROVIDER_SDK),
        invocation_apis=_values(application, CouplingKind.INVOCATION),
        model_identifiers=_values(application, CouplingKind.MODEL_IDENTIFIER),
        prompt_fields=_values(application, CouplingKind.PROMPT),
        parameters=parameters,
        streaming=any(item.kind is CouplingKind.STREAMING for item in application.findings),
        reasoning_controls=sorted(
            name
            for name in parameters
            if name in {"thinking", "reasoning", "reasoning_effort", "effort"}
        ),
        multimodal_inputs=_values(application, CouplingKind.MULTIMODAL),
        multimodal_contracts=application.multimodal_inputs,
        tool_fields=_values(application, CouplingKind.TOOL),
        structured_output_fields=_values(application, CouplingKind.STRUCTURED_OUTPUT),
        response_parsers=_values(application, CouplingKind.RESPONSE_PARSER),
        parser_assumptions=parser_assumptions,
        tool_definitions=application.tool_definitions,
        structured_outputs=application.structured_outputs,
        tool_compatibility=_tool_assessment(application, target) if target else None,
        structured_output_compatibility=(
            _structured_output_assessment(application, target) if target else None
        ),
        source_locations=all_locations,
        warnings=[
            *application.warnings,
            *(
                ["No supported LLM invocation was detected."]
                if not _values(application, CouplingKind.INVOCATION)
                else []
            ),
        ],
    )


# The platforms _target_invocation below has a deterministic adapter for, and
# the subset whose adapter exposes a structured-output request field. These are
# the single source for "can the toolkit drive this platform" gating (the
# blocker resolver imports them); keep them in lockstep with the branches of
# _target_invocation.
ADAPTER_PLATFORMS = frozenset({"anthropic-api", "openai-api", "amazon-bedrock"})
STRUCTURED_OUTPUT_FIELD_PLATFORMS = frozenset({"anthropic-api", "openai-api"})


def _target_invocation(target: ResolvedModel) -> TargetInvocation:
    platform = target.platform
    if platform is None:
        if len(target.platforms) != 1:
            raise ValueError(
                "target platform is required when a model has multiple representations"
            )
        platform = target.platforms[0]
    common = {
        "provider": target.identity.provider,
        "platform": platform.platform,
        "model_id": platform.model_id,
    }
    if platform.platform == "anthropic-api":
        return TargetInvocation(
            **common,
            sdk="anthropic",
            client="Anthropic",
            operation="messages.create",
            model_field="model",
            message_field="messages",
            system_field="system",
            tool_field="tools",
            structured_output_field="output_config",
            image_input_format="messages.content.image.source",
            document_input_format="messages.content.document.source",
        )
    if platform.platform == "openai-api":
        return TargetInvocation(
            **common,
            sdk="openai",
            client="OpenAI",
            operation="responses.create",
            model_field="model",
            message_field="input",
            system_field="instructions",
            tool_field="tools",
            structured_output_field="text.format",
            image_input_format="input.content.input_image",
            document_input_format="input.content.input_file",
        )
    if platform.platform == "amazon-bedrock":
        return TargetInvocation(
            **common,
            sdk="boto3",
            client="bedrock-runtime",
            operation="converse",
            model_field="modelId",
            message_field="messages",
            system_field="system",
            parameter_container="inferenceConfig",
            tool_field="toolConfig",
            structured_output_field=None,
            image_input_format="messages.content.image",
            document_input_format="messages.content.document",
        )
    return TargetInvocation(
        **common,
        sdk="unknown",
        client="unknown",
        operation="unknown",
        model_field="model",
        message_field="messages",
    )


def _registry_parameter_name(name: str, target: ResolvedModel) -> tuple[str, bool]:
    normalized = "max_tokens" if name == "max_output_tokens" else name
    provider = target.identity.provider
    if provider == "anthropic" and name == "reasoning_effort":
        for candidate in ("effort", "adaptive_thinking", "thinking"):
            if candidate in target.parameters:
                return candidate, True
    if provider == "openai" and name in {"effort", "thinking", "adaptive_thinking"}:
        return "reasoning_effort", True
    if name == "thinking" and "effort" in target.parameters:
        return "effort", True
    return normalized, False


def _target_parameter_location(registry_name: str, platform: str) -> tuple[str, str | None]:
    if platform == "openai-api":
        return ("max_output_tokens" if registry_name == "max_tokens" else registry_name, None)
    if platform == "anthropic-api":
        return registry_name, None
    if platform == "amazon-bedrock":
        common = {
            "max_tokens": "maxTokens",
            "temperature": "temperature",
            "top_p": "topP",
            "stop": "stopSequences",
        }
        if registry_name in common:
            return common[registry_name], "inferenceConfig"
        return registry_name, "additionalModelRequestFields"
    return registry_name, None


def _target_tool_choice(
    choice: str | None,
    *,
    platform: str,
    name: str | None,
) -> str | dict[str, Any] | None:
    if choice is None:
        return None
    if platform == "anthropic-api":
        mapping = {"auto": "auto", "any": "any", "required": "any", "none": "none"}
        kind = mapping.get(choice)
        if choice == "tool" and name:
            return {"type": "tool", "name": name}
        return {"type": kind} if kind else None
    if platform == "openai-api":
        if choice == "tool" and name:
            return {"type": "function", "name": name}
        return "required" if choice == "any" else choice
    if platform == "amazon-bedrock":
        if choice == "tool" and name:
            return {"tool": {"name": name}}
        if choice in {"any", "required"}:
            return {"any": {}}
        if choice == "auto":
            return {"auto": {}}
    return None


def _tool_schema_migrations(
    definitions: list[ToolDefinition],
    target: ResolvedModel,
    target_invocation: TargetInvocation,
) -> list[ToolSchemaMigration]:
    migrations: list[ToolSchemaMigration] = []
    capability = target.effective_capabilities.tool_use
    target_field = (
        "toolConfig.tools"
        if target_invocation.platform == "amazon-bedrock"
        and target_invocation.tool_field is not None
        else target_invocation.tool_field
    )
    target_choice_field = {
        "anthropic-api": "tool_choice",
        "openai-api": "tool_choice",
        "amazon-bedrock": "toolConfig.toolChoice",
    }.get(target_invocation.platform)
    for definition in definitions:
        candidate: dict[str, Any] | None
        notes = _schema_issues(definition.input_schema)
        schema_error = _schema_validation_error(definition.input_schema)
        if schema_error:
            notes.append(schema_error)
        if definition.strict is True:
            notes.append("Strict tool-schema semantics require target verification.")
        if capability is False or target_invocation.tool_field is None:
            state = CompatibilityState.UNSUPPORTED
            candidate = None
            notes.append("No supported target tool field is available.")
        elif (
            capability is None
            or not definition.name
            or definition.input_schema is None
            or schema_error is not None
        ):
            state = CompatibilityState.UNKNOWN
            candidate = None
        else:
            same_format = definition.provider_format.startswith(f"{target_invocation.platform}.")
            state = (
                CompatibilityState.SEMANTICALLY_DIFFERENT
                if notes
                else (
                    CompatibilityState.DIRECTLY_COMPATIBLE
                    if same_format
                    else CompatibilityState.MECHANICALLY_CONVERTIBLE
                )
            )
            if target_invocation.platform == "anthropic-api":
                candidate = {
                    "name": definition.name,
                    **(
                        {"description": definition.description}
                        if definition.description is not None
                        else {}
                    ),
                    "input_schema": definition.input_schema,
                }
            elif target_invocation.platform == "openai-api":
                candidate = {
                    "type": "function",
                    "name": definition.name,
                    **(
                        {"description": definition.description}
                        if definition.description is not None
                        else {}
                    ),
                    "parameters": definition.input_schema,
                    **({"strict": definition.strict} if definition.strict is not None else {}),
                }
            elif target_invocation.platform == "amazon-bedrock":
                candidate = {
                    "toolSpec": {
                        "name": definition.name,
                        **(
                            {"description": definition.description}
                            if definition.description is not None
                            else {}
                        ),
                        "inputSchema": {"json": definition.input_schema},
                    }
                }
            else:
                candidate = None
                state = CompatibilityState.UNSUPPORTED
                notes.append("No deterministic target tool wrapper is registered.")
        target_choice = _target_tool_choice(
            definition.choice,
            platform=target_invocation.platform,
            name=definition.name,
        )
        if definition.choice is not None and target_choice is None:
            notes.append(f"Tool-choice mode {definition.choice!r} has no deterministic mapping.")
            if state not in {CompatibilityState.UNSUPPORTED, CompatibilityState.UNKNOWN}:
                state = CompatibilityState.SEMANTICALLY_DIFFERENT
        migrations.append(
            ToolSchemaMigration(
                name=definition.name,
                source_format=definition.provider_format,
                target_platform=target_invocation.platform,
                target_field=target_field,
                state=state,
                source_location=definition.location,
                target_definition=candidate,
                target_choice_field=target_choice_field if target_choice is not None else None,
                target_choice=target_choice,
                review_notes=sorted(set(notes)),
            )
        )
    return migrations


def _structured_output_migrations(
    contracts: list[StructuredOutputContract],
    target: ResolvedModel,
    target_invocation: TargetInvocation,
) -> list[StructuredOutputMigration]:
    migrations: list[StructuredOutputMigration] = []
    capability = target.effective_capabilities.structured_output
    for contract in contracts:
        candidate: dict[str, Any] | None
        notes = _schema_issues(contract.json_schema)
        schema_error = _schema_validation_error(contract.json_schema)
        if schema_error:
            notes.append(schema_error)
        if contract.strict is True:
            notes.append("Strict structured-output semantics require target verification.")
        if capability is False or target_invocation.structured_output_field is None:
            state = CompatibilityState.UNSUPPORTED
            candidate = None
            notes.append("No supported target structured-output field is available.")
        elif capability is None or contract.json_schema is None or schema_error is not None:
            state = CompatibilityState.UNKNOWN
            candidate = None
        else:
            same_format = contract.provider_format.startswith(f"{target_invocation.platform}.")
            state = (
                CompatibilityState.SEMANTICALLY_DIFFERENT
                if notes
                else (
                    CompatibilityState.DIRECTLY_COMPATIBLE
                    if same_format
                    else CompatibilityState.MECHANICALLY_CONVERTIBLE
                )
            )
            if target_invocation.platform == "anthropic-api":
                candidate = {
                    "format": {
                        "type": "json_schema",
                        "schema": contract.json_schema,
                    }
                }
            elif target_invocation.platform == "openai-api":
                candidate = {
                    "type": "json_schema",
                    "name": contract.name or "response",
                    "schema": contract.json_schema,
                    **({"strict": contract.strict} if contract.strict is not None else {}),
                }
            else:
                candidate = None
                state = CompatibilityState.UNSUPPORTED
                notes.append("No deterministic target output wrapper is registered.")
        migrations.append(
            StructuredOutputMigration(
                name=contract.name,
                source_format=contract.provider_format,
                target_platform=target_invocation.platform,
                target_field=target_invocation.structured_output_field,
                state=state,
                source_location=contract.location,
                target_configuration=candidate,
                review_notes=sorted(set(notes)),
            )
        )
    return migrations


def _append_contract_review(
    label: str,
    state: CompatibilityState,
    notes: list[str],
    *,
    warnings: list[str],
    blockers: list[MigrationBlocker],
    capability: str,
    location: SourceLocation,
    capability_url: str | None,
) -> None:
    # Two same-named contracts in different files must stay distinct blockers,
    # so per-contract blockers discriminate by their source location.
    discriminator = f"{location.path}:{location.line}"
    if state is CompatibilityState.UNSUPPORTED:
        blockers.append(
            migration_blocker(
                code=f"{capability}_payload_unsupported",
                category=BlockerCategory.CAPABILITY,
                message=f"{label} has no safe target payload candidate.",
                evidence_urls=[capability_url] if capability_url else [],
                locations=[location],
                data={"capability": capability},
                discriminator=discriminator,
            )
        )
    for note in notes:
        message = f"{label} migration: {note}"
        if note.startswith("Invalid Draft 2020-12 JSON Schema"):
            blockers.append(
                migration_blocker(
                    code="invalid_json_schema",
                    category=BlockerCategory.INVALID_SCHEMA,
                    message=message,
                    locations=[location],
                    data={"capability": capability},
                    discriminator=discriminator,
                )
            )
        elif state in {
            CompatibilityState.UNKNOWN,
            CompatibilityState.SEMANTICALLY_DIFFERENT,
        }:
            warnings.append(message)


def _configuration_migration(
    application: ApplicationAnalysis,
    source: ResolvedModel,
    target: TargetInvocation,
) -> PlatformConfigurationMigration:
    locations = sorted(
        {
            finding.location
            for finding in application.findings
            if finding.kind in {CouplingKind.CONFIGURATION, CouplingKind.RETRY_ERROR_HANDLING}
        },
        key=lambda item: (item.path, item.line, item.column),
    )
    credential_strategy: Literal["environment_variable", "aws_default_chain", "unknown"]
    credential_variables: list[str]
    region_variables: list[str]
    review_notes: list[str] = []
    if target.platform == "anthropic-api":
        credential_strategy = "environment_variable"
        credential_variables = ["ANTHROPIC_API_KEY"]
        region_variables = []
    elif target.platform == "openai-api":
        credential_strategy = "environment_variable"
        credential_variables = ["OPENAI_API_KEY"]
        region_variables = []
    elif target.platform == "amazon-bedrock":
        credential_strategy = "aws_default_chain"
        credential_variables = []
        region_variables = ["AWS_REGION", "AWS_DEFAULT_REGION"]
        review_notes.append(
            "Use the normal AWS SDK credential chain; never persist AWS credential values."
        )
    else:
        credential_strategy = "unknown"
        credential_variables = []
        region_variables = []
        review_notes.append("No deterministic credential strategy is registered.")
    same_route = (
        application.requirements.source_providers == {source.identity.provider}
        and application.requirements.source_platforms == {target.platform}
        and source.identity.provider == target.provider
    )
    state = (
        CompatibilityState.UNSUPPORTED
        if credential_strategy == "unknown" or target.operation == "unknown"
        else (
            CompatibilityState.DIRECTLY_COMPATIBLE
            if same_route
            else CompatibilityState.MECHANICALLY_CONVERTIBLE
        )
    )
    required_changes = [
        f"Instantiate {target.sdk}.{target.client} for {target.platform}.",
    ]
    if credential_strategy == "environment_variable":
        required_changes.append(
            "Reference the target credential through " + ", ".join(credential_variables) + "."
        )
    elif credential_strategy == "aws_default_chain":
        required_changes.append(
            "Use the AWS SDK default credential chain and configure an explicit deployment region."
        )
    else:
        required_changes.append("Provide a reviewed target credential/configuration adapter.")
    return PlatformConfigurationMigration(
        source_providers=sorted(application.requirements.source_providers),
        source_platforms=sorted(application.requirements.source_platforms),
        target_provider=target.provider,
        target_platform=target.platform,
        target_sdk=target.sdk,
        target_client=target.client,
        credential_strategy=credential_strategy,
        credential_environment_variables=credential_variables,
        region_environment_variables=region_variables,
        state=state,
        source_locations=locations,
        required_changes=required_changes,
        review_notes=review_notes,
    )


def _finding_locations(
    application: ApplicationAnalysis,
    kind: CouplingKind,
    *,
    value: str | None = None,
    parameter: str | None = None,
) -> list[SourceLocation]:
    return sorted(
        {
            item.location
            for item in application.findings
            if item.kind is kind
            and (value is None or item.value == value)
            and (parameter is None or (item.metadata or {}).get("name") == parameter)
        },
        key=lambda item: (item.path, item.line, item.column),
    )


def prepare_invocation_migration(
    application: ApplicationAnalysis,
    source: ResolvedModel,
    target: ResolvedModel,
    *,
    preparation_warnings: list[str] | None = None,
    preparation_blockers: list[MigrationBlocker] | None = None,
) -> InvocationMigrationSpec:
    analysis = analyze_invocation(application, target)
    target_invocation = _target_invocation(target)
    tool_schema_migrations = _tool_schema_migrations(
        analysis.tool_definitions, target, target_invocation
    )
    structured_output_migrations = _structured_output_migrations(
        analysis.structured_outputs, target, target_invocation
    )
    configuration_migration = _configuration_migration(application, source, target_invocation)
    mappings: list[ParameterMapping] = []
    mapped_registry_parameters: set[str] = set()
    blockers: list[MigrationBlocker] = list(preparation_blockers or [])
    warnings = [*analysis.warnings, *(preparation_warnings or [])]

    def capability_url(field: str) -> str | None:
        return capability_evidence_url(target.profile, target.platform, field)

    if target_invocation.operation == "unknown":
        blockers.append(
            migration_blocker(
                code="no_invocation_adapter",
                category=BlockerCategory.CAPABILITY,
                message=(
                    f"No deterministic invocation adapter exists for {target_invocation.platform}."
                ),
                data={"platform": target_invocation.platform},
            )
        )
    for name, value in sorted(analysis.parameters.items()):
        registry_name, semantic_mapping = _registry_parameter_name(name, target)
        mapped_registry_parameters.add(registry_name)
        support = target.parameters.get(registry_name)
        mapped_name, mapped_container = _target_parameter_location(
            registry_name, target_invocation.platform
        )
        target_name: str | None = mapped_name
        target_container: str | None = mapped_container
        if support is None:
            state = CompatibilityState.UNKNOWN
            rationale = "Target registry has no support fact for this parameter."
            warnings.append(f"Unknown target support for parameter {name}.")
            target_name = None
            target_container = None
        elif support.state is ParameterState.UNSUPPORTED:
            state = CompatibilityState.UNSUPPORTED
            rationale = support.notes or f"Target marks {registry_name} {support.state.value}."
            blockers.append(
                migration_blocker(
                    code="parameter_unsupported",
                    category=BlockerCategory.PARAMETER,
                    message=f"Parameter {name} cannot be carried over unchanged: {rationale}",
                    evidence_urls=(
                        [url]
                        if (url := evidence_url(target.profile, support.sources, "parameters"))
                        else []
                    ),
                    locations=_finding_locations(
                        application, CouplingKind.PARAMETER, parameter=name
                    ),
                    data={"parameter": name, "registry_name": registry_name, "notes": rationale},
                )
            )
            target_name = None
            target_container = None
        elif support.state is ParameterState.DEPRECATED:
            state = CompatibilityState.SEMANTICALLY_DIFFERENT
            rationale = support.notes or f"Target marks {registry_name} deprecated."
            warnings.append(f"Parameter {name} targets deprecated behavior: {rationale}")
            target_name = None
            target_container = None
        else:
            state = (
                CompatibilityState.SEMANTICALLY_DIFFERENT
                if semantic_mapping
                else (
                    CompatibilityState.DIRECTLY_COMPATIBLE
                    if target_name == name and target_container is None
                    else CompatibilityState.MECHANICALLY_CONVERTIBLE
                )
            )
            rationale = (
                f"Map {name} to {registry_name}; reasoning-control semantics require review."
                if semantic_mapping
                else (
                    "Parameter name is unchanged."
                    if target_name == name and target_container is None
                    else f"Map {name} to {target_name} for {target_invocation.platform}."
                )
            )
            if semantic_mapping:
                warnings.append(rationale)
        mappings.append(
            ParameterMapping(
                source_name=name,
                target_name=target_name,
                target_container=target_container,
                value=value,
                state=state,
                rationale=rationale,
            )
        )

    missing_required_parameters = sorted(
        name
        for name, support in target.parameters.items()
        if support.state is ParameterState.REQUIRED and name not in mapped_registry_parameters
    )
    for name in missing_required_parameters:
        warnings.append(f"Target requires parameter {name}; no source value was detected.")

    for capability_field, assessment in (
        ("tool_use", analysis.tool_compatibility),
        ("structured_output", analysis.structured_output_compatibility),
    ):
        if assessment and assessment.state is CompatibilityState.UNSUPPORTED:
            blockers.append(
                migration_blocker(
                    code=f"{capability_field}_unsupported",
                    category=BlockerCategory.CAPABILITY,
                    message=assessment.rationale,
                    evidence_urls=([url] if (url := capability_url(capability_field)) else []),
                    locations=assessment.locations,
                    data={"capability": capability_field},
                )
            )
        elif assessment and assessment.state in {
            CompatibilityState.UNKNOWN,
            CompatibilityState.SEMANTICALLY_DIFFERENT,
        }:
            warnings.append(assessment.rationale)
    for tool_migration in tool_schema_migrations:
        _append_contract_review(
            f"Tool {tool_migration.name or '<unknown>'}",
            tool_migration.state,
            tool_migration.review_notes,
            warnings=warnings,
            blockers=blockers,
            capability="tool_use",
            location=tool_migration.source_location,
            capability_url=capability_url("tool_use"),
        )
    for output_migration in structured_output_migrations:
        _append_contract_review(
            f"Output {output_migration.name or '<unnamed>'}",
            output_migration.state,
            output_migration.review_notes,
            warnings=warnings,
            blockers=blockers,
            capability="structured_output",
            location=output_migration.source_location,
            capability_url=capability_url("structured_output"),
        )
    if analysis.streaming and target.effective_capabilities.streaming is False:
        blockers.append(
            migration_blocker(
                code="streaming_unsupported",
                category=BlockerCategory.CAPABILITY,
                message=(
                    "The source streams responses, but the target declares streaming unsupported."
                ),
                evidence_urls=[url] if (url := capability_url("streaming")) else [],
                locations=_finding_locations(application, CouplingKind.STREAMING),
                data={"capability": "streaming"},
            )
        )
    elif analysis.streaming and target.effective_capabilities.streaming is None:
        warnings.append("The source streams responses; target streaming support is unknown.")
    for modality in analysis.multimodal_inputs:
        capability = getattr(target.effective_capabilities, modality, None)
        if capability is False:
            blockers.append(
                migration_blocker(
                    code=f"{modality}_unsupported",
                    category=BlockerCategory.CAPABILITY,
                    message=f"The target does not support required {modality}.",
                    evidence_urls=[url] if (url := capability_url(modality)) else [],
                    locations=_finding_locations(
                        application, CouplingKind.MULTIMODAL, value=modality
                    ),
                    data={"capability": modality},
                )
            )
        elif capability is None:
            warnings.append(f"Target support for required {modality} is unknown.")

    source_platforms = set(analysis.platforms)
    required_changes = [
        f"Replace model identifier with {target_invocation.model_id} in field "
        f"{target_invocation.model_field}.",
    ]
    required_changes.extend(
        f"Add required target parameter {name} with a reviewed value."
        for name in missing_required_parameters
    )
    if target_invocation.operation != "unknown":
        required_changes.append(
            f"Use {target_invocation.sdk}.{target_invocation.client} and "
            f"{target_invocation.operation}."
        )
    if source_platforms != {target_invocation.platform}:
        required_changes.append(
            f"Translate request shape to the {target_invocation.platform} message contract."
        )
    if analysis.tool_compatibility:
        tool_field = target_invocation.tool_field or "a supported target mechanism"
        required_changes.append(f"Translate tool configuration to {tool_field}.")
    if analysis.structured_output_compatibility:
        required_changes.append(
            "Preserve the output schema and revalidate downstream parser strictness."
        )
        if target_invocation.structured_output_field is None:
            blockers.append(
                migration_blocker(
                    code="structured_output_mapping_missing",
                    category=BlockerCategory.CAPABILITY,
                    message=(
                        f"No deterministic structured-output configuration mapping exists for "
                        f"{target_invocation.platform}."
                    ),
                    locations=analysis.structured_output_compatibility.locations,
                    data={
                        "capability": "structured_output",
                        "platform": target_invocation.platform,
                    },
                )
            )
    for contract in analysis.multimodal_contracts:
        target_format = (
            target_invocation.image_input_format
            if contract.modality == "image_input"
            else target_invocation.document_input_format
        )
        if target_format is None:
            blockers.append(
                migration_blocker(
                    code=f"{contract.modality}_mapping_missing",
                    category=BlockerCategory.CAPABILITY,
                    message=(
                        f"No deterministic {contract.modality} payload mapping exists for "
                        f"{target_invocation.platform}."
                    ),
                    locations=[contract.location],
                    data={
                        "capability": contract.modality,
                        "platform": target_invocation.platform,
                    },
                )
            )
        elif contract.provider_format != target_format:
            required_changes.append(
                f"Translate {contract.modality} payload from {contract.provider_format} "
                f"to {target_format}."
            )
    if source.identity.provider != target.identity.provider or source_platforms != {
        target_invocation.platform
    }:
        required_changes.append(
            "Replace provider/platform authentication, configuration, and error-handling "
            "assumptions."
        )
    return InvocationMigrationSpec(
        source_model=source.canonical_name,
        target_model=target.canonical_name,
        source_analysis=analysis,
        target=target_invocation,
        parameter_mappings=mappings,
        tool_schema_migrations=tool_schema_migrations,
        structured_output_migrations=structured_output_migrations,
        configuration_migration=configuration_migration,
        required_changes=required_changes,
        warnings=sorted(set(warnings)),
        blockers=dedupe_blockers(blockers),
    )
