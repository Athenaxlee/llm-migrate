"""Narrow, extensible AST scanner for Python LLM applications."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Literal

from llm_migrate.core.models import (
    ApplicationAnalysis,
    ApplicationFinding,
    ApplicationRequirements,
    CouplingKind,
    MultimodalInputContract,
    SourceLocation,
    StructuredOutputContract,
    ToolDefinition,
)

_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
_PARAMETERS = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_output_tokens",
    "reasoning_effort",
    "thinking",
    "effort",
    "stop",
    "seed",
}
_ENV_MARKERS = ("ANTHROPIC", "OPENAI", "AWS", "BEDROCK", "MODEL")


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal(node: ast.AST | None) -> str | bool | int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool, int, float)):
        return node.value
    return None


def _structured_literal(node: ast.AST | None) -> Any | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, list):
        list_result = [_structured_value(item) for item in value]
        return list_result if all(item is not _UNSUPPORTED for item in list_result) else None
    if isinstance(value, tuple):
        tuple_result = [_structured_value(item) for item in value]
        return tuple_result if all(item is not _UNSUPPORTED for item in tuple_result) else None
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        dict_result = {key: _structured_value(item) for key, item in value.items()}
        return (
            dict_result if all(item is not _UNSUPPORTED for item in dict_result.values()) else None
        )
    return None


_UNSUPPORTED = object()


def _structured_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        sequence_result = [_structured_value(item) for item in value]
        return (
            sequence_result
            if all(item is not _UNSUPPORTED for item in sequence_result)
            else _UNSUPPORTED
        )
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        mapping_result = {key: _structured_value(item) for key, item in value.items()}
        return (
            mapping_result
            if all(item is not _UNSUPPORTED for item in mapping_result.values())
            else _UNSUPPORTED
        )
    return _UNSUPPORTED


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.findings: list[ApplicationFinding] = []
        self.aliases: dict[str, str] = {}
        self.clients: dict[str, tuple[str | None, str | None]] = {}
        self.constants: dict[str, str | bool | int | float] = {}
        self.objects: dict[str, Any] = {}
        self.tool_definitions: list[ToolDefinition] = []
        self.structured_outputs: list[StructuredOutputContract] = []
        self.multimodal_inputs: list[MultimodalInputContract] = []

    def _value(self, node: ast.AST | None) -> str | bool | int | float | None:
        value = _literal(node)
        if value is None and isinstance(node, ast.Name):
            return self.constants.get(node.id)
        return value

    def _object(self, node: ast.AST | None) -> Any | None:
        if isinstance(node, ast.Name):
            return self.objects.get(node.id)
        return _structured_literal(node)

    def _location(self, node: ast.AST) -> SourceLocation:
        return SourceLocation(
            path=self.path,
            line=getattr(node, "lineno", 1),
            column=getattr(node, "col_offset", 0),
            end_line=getattr(node, "end_lineno", None),
        )

    def _add(
        self,
        node: ast.AST,
        kind: CouplingKind,
        detail: str,
        *,
        provider: str | None = None,
        platform: str | None = None,
        value: str | bool | int | float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.findings.append(
            ApplicationFinding(
                kind=kind,
                provider=provider,
                platform=platform,
                value=value,
                detail=detail,
                location=self._location(node),
                metadata=metadata,
            )
        )

    def visit_Import(self, node: ast.Import) -> Any:
        for name in node.names:
            root = name.name.split(".")[0]
            alias = name.asname or root
            self.aliases[alias] = name.name
            if root in {"anthropic", "openai", "boto3"}:
                provider, platform = _provider_platform(root)
                self._add(
                    node,
                    CouplingKind.PROVIDER_SDK,
                    f"imports {name.name}",
                    provider=provider,
                    platform=platform,
                    value=name.name,
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        root = (node.module or "").split(".")[0]
        for name in node.names:
            self.aliases[name.asname or name.name] = f"{node.module}.{name.name}"
        if root in {"anthropic", "openai", "boto3"}:
            provider, platform = _provider_platform(root)
            self._add(
                node,
                CouplingKind.PROVIDER_SDK,
                f"imports from {node.module}",
                provider=provider,
                platform=platform,
                value=node.module,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        assigned_value = _literal(node.value)
        if assigned_value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.constants[target.id] = assigned_value
        assigned_object = _structured_literal(node.value)
        if assigned_object is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.objects[target.id] = assigned_object
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and any(marker in target.id.casefold() for marker in ("prompt", "system"))
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and _dotted(node.value.func).endswith("read_text")
            ):
                file_value = node.value.func.value
                if isinstance(file_value, ast.Call) and file_value.args:
                    prompt_path = _literal(file_value.args[0])
                    if isinstance(prompt_path, str):
                        self._add(
                            node,
                            CouplingKind.PROMPT,
                            "loads prompt content from a file",
                            value=prompt_path,
                        )
        if isinstance(node.value, ast.Call):
            call_name = _dotted(node.value.func)
            root = call_name.split(".")[0]
            expanded = call_name.replace(root, self.aliases.get(root, root), 1).casefold()
            context = _provider_platform_from_call(expanded)
            if (
                expanded.endswith("boto3.client")
                and node.value.args
                and _literal(node.value.args[0]) == "bedrock-runtime"
            ):
                context = (None, "amazon-bedrock")
            if context != (None, None):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.clients[target.id] = context
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        call_name = _dotted(node.func)
        root = call_name.split(".")[0]
        expanded = call_name.replace(root, self.aliases.get(root, root), 1)
        lowered = expanded.casefold()
        provider, platform = _provider_platform_from_call(lowered)
        if root in self.clients:
            provider, platform = self.clients[root]
        if any(
            marker in lowered
            for marker in (
                "messages.create",
                "messages.stream",
                "responses.create",
                "responses.stream",
                "chat.completions.create",
                "invoke_model",
                ".converse",
            )
        ):
            self._add(
                node,
                CouplingKind.INVOCATION,
                f"calls {call_name}",
                provider=provider,
                platform=platform,
                value=call_name,
            )
        keyword_map = {item.arg: item.value for item in node.keywords if item.arg}
        model_node = keyword_map.get("model") or keyword_map.get("modelId")
        model_value = self._value(model_node)
        if isinstance(model_value, str):
            if model_value.startswith("anthropic."):
                provider = "anthropic"
                platform = platform or "amazon-bedrock"
            self._add(
                model_node or node,
                CouplingKind.MODEL_IDENTIFIER,
                "passes a literal model identifier",
                provider=provider,
                platform=platform,
                value=model_value,
            )
        for name in sorted(_PARAMETERS & keyword_map.keys()):
            value = self._value(keyword_map[name])
            self._add(
                keyword_map[name],
                CouplingKind.PARAMETER,
                f"sets generation parameter {name}",
                provider=provider,
                platform=platform,
                value=name if value is None else f"{name}={value}",
                metadata={"name": name, "value": value},
            )
        tool_choice = self._object(keyword_map.get("tool_choice"))
        for name in ("tools", "functions", "toolConfig"):
            if name in keyword_map:
                raw_tools = self._object(keyword_map[name])
                definitions = _normalize_tool_definitions(
                    raw_tools,
                    provider=provider,
                    platform=platform,
                    field=name,
                    choice=tool_choice,
                    location=self._location(keyword_map[name]),
                )
                self.tool_definitions.extend(definitions)
                self._add(
                    keyword_map[name],
                    CouplingKind.TOOL,
                    f"configures {name}",
                    provider=provider,
                    platform=platform,
                    value=name,
                    metadata={"definition_count": len(definitions)},
                )
        if "tool_choice" in keyword_map:
            self._add(
                keyword_map["tool_choice"],
                CouplingKind.TOOL,
                "configures tool_choice",
                provider=provider,
                platform=platform,
                value="tool_choice",
                metadata={"choice": tool_choice},
            )
        for name in ("response_format", "output_config", "text"):
            if name in keyword_map and (name != "text" or isinstance(keyword_map[name], ast.Dict)):
                raw_output = self._object(keyword_map[name])
                contract = _normalize_structured_output(
                    raw_output,
                    provider=provider,
                    platform=platform,
                    field=name,
                    location=self._location(keyword_map[name]),
                )
                if contract is not None:
                    self.structured_outputs.append(contract)
                self._add(
                    keyword_map[name],
                    CouplingKind.STRUCTURED_OUTPUT,
                    f"configures structured output through {name}",
                    provider=provider,
                    platform=platform,
                    value=name,
                    metadata={"schema_detected": bool(contract and contract.schema)},
                )
        if self._value(keyword_map.get("stream")) is True:
            self._add(
                keyword_map["stream"],
                CouplingKind.STREAMING,
                "enables streaming",
                provider=provider,
                platform=platform,
                value=True,
            )
        if any(
            marker in lowered
            for marker in (
                "messages.stream",
                "responses.stream",
                "converse_stream",
                "response_stream",
            )
        ):
            self._add(
                node,
                CouplingKind.STREAMING,
                f"uses streaming invocation {call_name}",
                provider=provider,
                platform=platform,
                value=True,
            )
        for name in ("system", "messages", "input", "prompt"):
            if name in keyword_map:
                self._add(
                    keyword_map[name],
                    CouplingKind.PROMPT,
                    f"supplies prompt content through {name}",
                    provider=provider,
                    platform=platform,
                    value=name,
                )
        if any(marker in lowered for marker in ("json.loads", "model_validate_json", ".parse")):
            self._add(
                node,
                CouplingKind.RESPONSE_PARSER,
                f"parses a model response with {call_name}",
                value=call_name,
            )
        if any(marker in lowered for marker in ("os.getenv", "os.environ.get")):
            key = _literal(node.args[0] if node.args else None)
            if isinstance(key, str) and any(marker in key.upper() for marker in _ENV_MARKERS):
                self._add(
                    node,
                    CouplingKind.CONFIGURATION,
                    f"reads LLM configuration {key}",
                    value=key,
                )
        if lowered.endswith("boto3.client") and node.args:
            service = _literal(node.args[0])
            if service == "bedrock-runtime":
                self._add(
                    node,
                    CouplingKind.CONFIGURATION,
                    "creates an Amazon Bedrock Runtime client",
                    provider="anthropic",
                    platform="amazon-bedrock",
                    value=service,
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if _dotted(node.value).casefold() == "os.environ":
            key = self._value(node.slice)
            if isinstance(key, str) and any(marker in key.upper() for marker in _ENV_MARKERS):
                self._add(
                    node,
                    CouplingKind.CONFIGURATION,
                    f"reads LLM configuration {key}",
                    value=key,
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        exception_name = _dotted(node.type) if node.type else ""
        if any(
            marker in exception_name.casefold()
            for marker in ("apierror", "ratelimit", "timeout", "clienterror")
        ):
            self._add(
                node,
                CouplingKind.RETRY_ERROR_HANDLING,
                f"handles provider invocation error {exception_name}",
                value=exception_name,
            )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> Any:
        pairs = [
            (_literal(key), _literal(value))
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        ]
        image_marker = any(
            key in {"image", "image_url"} or (key == "type" and value in {"image", "input_image"})
            for key, value in pairs
        )
        document_marker = any(
            key == "document" or (key == "type" and value in {"document", "input_file"})
            for key, value in pairs
        )
        if image_marker:
            image_type = next((value for key, value in pairs if key == "type"), None)
            image_source_kind: Literal["url", "base64", "bytes", "file", "unknown"] = (
                "url" if any(key == "image_url" for key, _ in pairs) else "unknown"
            )
            self.multimodal_inputs.append(
                MultimodalInputContract(
                    modality="image_input",
                    provider_format=(
                        "input.content.input_image"
                        if image_type == "input_image"
                        else "messages.content.image.source"
                        if image_type == "image"
                        else "unknown"
                    ),
                    source_kind=image_source_kind,
                    location=self._location(node),
                )
            )
            self._add(
                node,
                CouplingKind.MULTIMODAL,
                "supplies image input",
                value="image_input",
            )
        if document_marker:
            document_source_kind: Literal["url", "base64", "bytes", "file", "unknown"] = "unknown"
            for key, value_node in zip(node.keys, node.values, strict=True):
                if _literal(key) != "document" or not isinstance(value_node, ast.Dict):
                    continue
                for nested_key, nested_value in zip(
                    value_node.keys, value_node.values, strict=True
                ):
                    if _literal(nested_key) == "source" and isinstance(nested_value, ast.Dict):
                        nested_pairs = [
                            (_literal(item_key), item_value)
                            for item_key, item_value in zip(
                                nested_value.keys, nested_value.values, strict=True
                            )
                            if item_key is not None
                        ]
                        if any(key == "bytes" for key, _ in nested_pairs):
                            document_source_kind = "bytes"
            self.multimodal_inputs.append(
                MultimodalInputContract(
                    modality="document_input",
                    provider_format="messages.content.document",
                    source_kind=document_source_kind,
                    location=self._location(node),
                )
            )
            self._add(
                node,
                CouplingKind.MULTIMODAL,
                "supplies document input",
                value="document_input",
            )
        self.generic_visit(node)


def _provider_format(provider: str | None, platform: str | None, field: str) -> str:
    return f"{platform or provider or 'unknown'}.{field}"


def _choice_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("type"), str):
            return str(value["type"])
        if len(value) == 1:
            return str(next(iter(value)))
    return None


def _normalize_tool_definitions(
    value: Any,
    *,
    provider: str | None,
    platform: str | None,
    field: str,
    choice: Any,
    location: SourceLocation,
) -> list[ToolDefinition]:
    if field == "toolConfig" and isinstance(value, dict):
        choice = value.get("toolChoice", choice)
        value = value.get("tools")
    if not isinstance(value, list):
        return []
    definitions: list[ToolDefinition] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        item = raw
        if isinstance(item.get("function"), dict):
            item = item["function"]
        elif isinstance(item.get("toolSpec"), dict):
            item = item["toolSpec"]
        schema = item.get("input_schema", item.get("parameters", item.get("inputSchema")))
        if isinstance(schema, dict) and isinstance(schema.get("json"), dict):
            schema = schema["json"]
        definitions.append(
            ToolDefinition(
                name=item.get("name") if isinstance(item.get("name"), str) else None,
                description=(
                    item.get("description") if isinstance(item.get("description"), str) else None
                ),
                input_schema=schema if isinstance(schema, dict) else None,
                strict=item.get("strict") if isinstance(item.get("strict"), bool) else None,
                provider_format=_provider_format(provider, platform, field),
                choice=_choice_name(choice),
                location=location,
            )
        )
    return definitions


def _normalize_structured_output(
    value: Any,
    *,
    provider: str | None,
    platform: str | None,
    field: str,
    location: SourceLocation,
) -> StructuredOutputContract | None:
    if not isinstance(value, dict):
        return StructuredOutputContract(
            provider_format=_provider_format(provider, platform, field),
            location=location,
        )
    item = value
    if isinstance(item.get("format"), dict):
        item = item["format"]
    if isinstance(item.get("json_schema"), dict):
        item = item["json_schema"]
    schema = item.get("schema")
    return StructuredOutputContract(
        name=item.get("name") if isinstance(item.get("name"), str) else None,
        json_schema=schema if isinstance(schema, dict) else None,
        strict=item.get("strict") if isinstance(item.get("strict"), bool) else None,
        provider_format=_provider_format(provider, platform, field),
        location=location,
    )


def _provider_platform(root: str) -> tuple[str | None, str | None]:
    return {
        "anthropic": ("anthropic", "anthropic-api"),
        "openai": ("openai", "openai-api"),
        "boto3": (None, "amazon-bedrock"),
    }.get(root, (None, None))


def _provider_platform_from_call(call: str) -> tuple[str | None, str | None]:
    if "anthropic" in call:
        return "anthropic", "anthropic-api"
    if "openai" in call:
        return "openai", "openai-api"
    if "bedrock" in call or "invoke_model" in call or ".converse" in call:
        return None, "amazon-bedrock"
    return None, None


def _requirements(findings: list[ApplicationFinding]) -> ApplicationRequirements:
    capabilities: set[str] = set()
    parameters: set[str] = set()
    models: set[str] = set()
    providers: set[str] = set()
    platforms: set[str] = set()
    for finding in findings:
        if finding.kind is CouplingKind.TOOL:
            capabilities.add("tool_use")
        elif finding.kind is CouplingKind.STRUCTURED_OUTPUT:
            capabilities.add("structured_output")
        elif finding.kind is CouplingKind.STREAMING:
            capabilities.add("streaming")
        elif finding.kind is CouplingKind.MULTIMODAL and isinstance(finding.value, str):
            capabilities.add(finding.value)
        elif finding.kind is CouplingKind.PARAMETER and isinstance(finding.value, str):
            parameters.add(finding.value.split("=", 1)[0])
        elif finding.kind is CouplingKind.MODEL_IDENTIFIER and isinstance(finding.value, str):
            models.add(finding.value)
        if finding.provider:
            providers.add(finding.provider)
        if finding.platform:
            platforms.add(finding.platform)
    return ApplicationRequirements(
        required_capabilities=capabilities,
        required_parameters=parameters,
        source_models=models,
        source_providers=providers,
        source_platforms=platforms,
    )


def scan_application(root: Path | str) -> ApplicationAnalysis:
    path = Path(root)
    if not path.exists():
        raise ValueError(f"application path does not exist: {path}")
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    files = [
        item
        for item in files
        if not any(
            part in _IGNORED_DIRECTORIES
            for part in item.relative_to(path if path.is_dir() else path.parent).parts
        )
    ]
    findings: list[ApplicationFinding] = []
    tool_definitions: list[ToolDefinition] = []
    structured_outputs: list[StructuredOutputContract] = []
    multimodal_inputs: list[MultimodalInputContract] = []
    warnings: list[str] = []
    base = path if path.is_dir() else path.parent
    for file_path in files:
        relative = file_path.relative_to(base).as_posix()
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            warnings.append(f"{relative}: {exc}")
            continue
        visitor = _Visitor(relative)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        tool_definitions.extend(visitor.tool_definitions)
        structured_outputs.extend(visitor.structured_outputs)
        multimodal_inputs.extend(visitor.multimodal_inputs)
    findings.sort(
        key=lambda item: (item.location.path, item.location.line, item.kind.value, item.detail)
    )
    return ApplicationAnalysis(
        root=str(path),
        files_scanned=len(files),
        findings=findings,
        requirements=_requirements(findings),
        tool_definitions=sorted(
            tool_definitions,
            key=lambda item: (item.location.path, item.location.line, item.name or ""),
        ),
        structured_outputs=sorted(
            structured_outputs,
            key=lambda item: (item.location.path, item.location.line, item.name or ""),
        ),
        multimodal_inputs=sorted(
            multimodal_inputs,
            key=lambda item: (item.location.path, item.location.line, item.modality),
        ),
        warnings=warnings,
    )
