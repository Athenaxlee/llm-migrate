"""Narrow, extensible AST scanner for Python LLM applications."""

from __future__ import annotations

import ast
import posixpath
from collections.abc import Sequence
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
from llm_migrate.core.prompt_documents import (
    PROMPT_SOURCE_SUFFIXES,
    STRUCTURED_SUFFIXES,
    extract_prompt_components,
    source_format,
)
from llm_migrate.scanners.config import (
    ConfigDocument,
    find_config_couplings,
    load_config_documents,
    lookup,
    resolve_reference,
)
from llm_migrate.scanners.provenance import (
    assemble_prompt_sources,
    summarize_prompt_discovery,
)

_IGNORED_DIRECTORIES = {
    ".git",
    ".llm-migrate",
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
_DOCUMENT_LOADERS = {
    "yaml.safe_load",
    "yaml.load",
    "json.load",
    "json.loads",
    "tomllib.load",
    "tomllib.loads",
}
_PATH_CONSTRUCTORS = {"Path", "pathlib.Path", "pathlib.PurePath", "pathlib.PurePosixPath"}
_LOADER_CALL_MARKERS = ("load", "read", "open", "fetch", "get", "prompt")


def _normalize_relative(value: str) -> str | None:
    """Normalize a candidate relative path; None for absolute or escaping paths."""
    if not value or value != value.strip() or "\n" in value:
        return None
    if value.startswith(("/", "~")) or (len(value) > 1 and value[1] == ":"):
        return None
    normalized = posixpath.normpath(value.replace("\\", "/"))
    return None if normalized.startswith("..") else normalized


def _subscript_chain(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    """Decompose `name["a"]["b"]` into ("name", ("a", "b")); None otherwise."""
    keys: list[str] = []
    current = node
    while isinstance(current, ast.Subscript):
        key = _literal(current.slice)
        if not isinstance(key, str):
            return None
        keys.append(key)
        current = current.value
    if isinstance(current, ast.Name) and keys:
        return current.id, tuple(reversed(keys))
    return None


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
    def __init__(
        self,
        path: str,
        catalog: dict[str, ConfigDocument] | None = None,
        known_files: set[str] | None = None,
    ) -> None:
        self.path = path
        self.catalog = catalog or {}
        self.known_files = known_files or set()
        self.findings: list[ApplicationFinding] = []
        self.aliases: dict[str, str] = {}
        self.clients: dict[str, tuple[str | None, str | None]] = {}
        self.constants: dict[str, str | bool | int | float] = {}
        self.objects: dict[str, Any] = {}
        self.tool_definitions: list[ToolDefinition] = []
        self.structured_outputs: list[StructuredOutputContract] = []
        self.multimodal_inputs: list[MultimodalInputContract] = []
        # Bounded local propagation state for prompt provenance discovery.
        self.path_values: dict[str, str] = {}
        self.file_handles: dict[str, str] = {}
        self.document_vars: dict[str, tuple[str, tuple[str, ...]]] = {}
        self.prompt_document_vars: dict[str, str] = {}
        self.prompt_text_vars: dict[str, str] = {}
        self.prompt_source_facts: dict[str, list[str]] = {}
        # Config documents whose values were traced into an invocation call
        # chain; they anchor semantic config couplings (V1.5.0-b).
        self.config_invocation_usage: set[str] = set()
        self.notes: list[str] = []

    def _expand(self, node: ast.AST) -> str:
        name = _dotted(node)
        root = name.split(".")[0]
        return name.replace(root, self.aliases.get(root, root), 1)

    def _static_path(self, node: ast.AST | None) -> str | None:
        """Resolve common deterministic path expressions to a relative path."""
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _normalize_relative(node.value)
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return self.path
            if node.id in self.path_values:
                return self.path_values[node.id]
            value = self.constants.get(node.id)
            return _normalize_relative(value) if isinstance(value, str) else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self._static_path(node.left)
            right = self._static_path(node.right)
            if left is not None and right is not None:
                return _normalize_relative(posixpath.join(left, right))
            return None
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            base = self._static_path(node.value)
            return posixpath.dirname(base) if base is not None else None
        if isinstance(node, ast.Call) and self._expand(node.func) in _PATH_CONSTRUCTORS:
            parts = [self._static_path(arg) for arg in node.args]
            if parts and all(part is not None for part in parts):
                joined = posixpath.join(*(part for part in parts if part is not None))
                return _normalize_relative(joined)
        return None

    def _open_path(self, call: ast.Call) -> str | None:
        """Path opened by `open(p)` or `<path expression>.open()`, if static."""
        if _dotted(call.func) == "open" and call.args:
            return self._static_path(call.args[0])
        if isinstance(call.func, ast.Attribute) and call.func.attr == "open":
            return self._static_path(call.func.value)
        return None

    def _loaded_document_path(self, call: ast.Call) -> str | None:
        """File behind `yaml.safe_load(...)`-style structured loads, if static."""
        if self._expand(call.func) not in _DOCUMENT_LOADERS or not call.args:
            return None
        argument = call.args[0]
        if isinstance(argument, ast.Name):
            return self.file_handles.get(argument.id)
        if isinstance(argument, ast.Call):
            opened = self._open_path(argument)
            if opened is not None:
                return opened
            if isinstance(argument.func, ast.Attribute) and argument.func.attr == "read_text":
                return self._static_path(argument.func.value)
        return None

    def _register_prompt_source(
        self, path: str, provenance: list[str], *, prompt_hint: bool
    ) -> bool:
        """Record a discovered prompt source when the file qualifies as one."""
        if path in self.prompt_source_facts:
            return True
        if source_format(path) is None or path not in self.known_files:
            return False
        document = self.catalog.get(path)
        if document is not None:
            if not extract_prompt_components(document.data):
                return False
        elif "." + path.rsplit(".", 1)[-1].casefold() in STRUCTURED_SUFFIXES:
            return False  # Structured file that failed to parse; already warned.
        elif not prompt_hint and "prompt" not in path.casefold():
            return False
        self.prompt_source_facts[path] = provenance
        return True

    def _track_config_access(
        self, node: ast.Assign, targets: list[str], chain: tuple[str, tuple[str, ...]]
    ) -> None:
        base_name, keys = chain
        if base_name not in self.document_vars:
            return
        document_path, prefix = self.document_vars[base_name]
        key_path = (*prefix, *keys)
        for name in targets:
            self.document_vars[name] = (document_path, key_path)
        document = self.catalog.get(document_path)
        value = lookup(document, key_path) if document is not None else None
        if not isinstance(value, str):
            return
        resolved = resolve_reference(document_path, value, self.known_files)
        if resolved is None:
            return
        for name in targets:
            self.path_values[name] = resolved
        registered = self._register_prompt_source(
            resolved,
            [
                f"{self.path} loads {document_path}",
                f"{document_path}: {'.'.join(key_path)} -> {resolved}",
            ],
            prompt_hint=any("prompt" in key.casefold() for key in key_path),
        )
        if registered:
            self._add(
                node,
                CouplingKind.PROMPT,
                "resolves a prompt file path from configuration",
                value=resolved,
                metadata={"config": document_path, "key_path": ".".join(key_path)},
            )

    def _track_loads(self, node: ast.Assign, targets: list[str], call: ast.Call) -> None:
        opened = self._open_path(call)
        if opened is not None:
            for name in targets:
                self.file_handles[name] = opened
        loaded = self._loaded_document_path(call)
        if loaded is not None and loaded in self.catalog:
            for name in targets:
                self.document_vars[name] = (loaded, ())
            if self._register_prompt_source(
                loaded, [f"{self.path} loads {loaded}"], prompt_hint=False
            ):
                for name in targets:
                    self.prompt_document_vars[name] = loaded
            return
        if isinstance(call.func, ast.Attribute) and call.func.attr == "read_text":
            file_path = self._static_path(call.func.value)
            promptish = any(
                marker in name.casefold() for name in targets for marker in ("prompt", "system")
            )
            if file_path is None or not (promptish or "prompt" in file_path.casefold()):
                return
            if self._register_prompt_source(
                file_path, [f"{self.path} loads {file_path}"], prompt_hint=True
            ):
                for name in targets:
                    self.prompt_text_vars[name] = file_path
                self._add(
                    node,
                    CouplingKind.PROMPT,
                    "loads prompt content from a file",
                    value=file_path,
                )
            elif file_path not in self.known_files and promptish:
                self.notes.append(
                    f"{self.path}: prompt file {file_path!r} was not found in the application."
                )
            return
        callee = _dotted(call.func).casefold()
        if any(marker in callee for marker in _LOADER_CALL_MARKERS):
            for argument in call.args:
                argument_path = self._static_path(argument)
                if argument_path is None:
                    continue
                if argument_path in self.prompt_source_facts or self._register_prompt_source(
                    argument_path,
                    [f"{self.path} loads {argument_path}"],
                    prompt_hint=False,
                ):
                    for name in targets:
                        self.prompt_document_vars[name] = argument_path
                    break

    def _classify_prompt_value(self, node: ast.AST) -> tuple[str, list[str]]:
        """Classify a prompt argument as inline, source-backed, or dynamic."""
        traced = {
            self.prompt_text_vars.get(child.id, self.prompt_document_vars.get(child.id))
            for child in ast.walk(node)
            if isinstance(child, ast.Name)
            and (child.id in self.prompt_text_vars or child.id in self.prompt_document_vars)
        }
        source_paths = sorted(path for path in traced if path is not None)
        if source_paths:
            return "source", source_paths
        dynamic_markers = (
            ast.Call,
            ast.Attribute,
            ast.Subscript,
            ast.Starred,
            ast.FormattedValue,
            ast.Await,
            ast.Lambda,
        )
        for child in ast.walk(node):
            if isinstance(child, dynamic_markers):
                return "dynamic", []
            if (
                isinstance(child, ast.Name)
                and child.id not in self.constants
                and child.id not in self.objects
            ):
                return "dynamic", []
        return "inline", []

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
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        static_path = self._static_path(node.value)
        if static_path is not None:
            for name in targets:
                self.path_values[name] = static_path
        chain = _subscript_chain(node.value)
        if chain is not None:
            self._track_config_access(node, targets, chain)
        if isinstance(node.value, ast.Call):
            self._track_loads(node, targets, node.value)
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

    def visit_With(self, node: ast.With) -> Any:
        for item in node.items:
            if isinstance(item.context_expr, ast.Call) and isinstance(item.optional_vars, ast.Name):
                opened = self._open_path(item.context_expr)
                if opened is not None:
                    self.file_handles[item.optional_vars.id] = opened
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
            dynamic_request = any(item.arg is None for item in node.keywords)
            self._add(
                node,
                CouplingKind.INVOCATION,
                f"calls {call_name}",
                provider=provider,
                platform=platform,
                value=call_name,
                metadata={"dynamic_request": True} if dynamic_request else None,
            )
            # Config documents whose values reach this invocation are anchored:
            # a name assigned from a loaded document, or a direct subscript
            # chain into one (the same loader recognition prompt discovery uses).
            for keyword in node.keywords:
                for child in ast.walk(keyword.value):
                    if isinstance(child, ast.Name) and child.id in self.document_vars:
                        self.config_invocation_usage.add(self.document_vars[child.id][0])
                    chain = _subscript_chain(child)
                    if chain is not None and chain[0] in self.document_vars:
                        self.config_invocation_usage.add(self.document_vars[chain[0]][0])
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
                resolution, source_paths = self._classify_prompt_value(keyword_map[name])
                self._add(
                    keyword_map[name],
                    CouplingKind.PROMPT,
                    f"supplies prompt content through {name}",
                    provider=provider,
                    platform=platform,
                    value=name,
                    metadata={"resolution": resolution, "sources": source_paths},
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


_CONFIG_COUPLING_DETAILS: dict[str, tuple[CouplingKind, str]] = {
    "model_id": (CouplingKind.MODEL_IDENTIFIER, "configures a model identifier"),
    "sampling": (CouplingKind.PARAMETER, "configures a sampling parameter"),
    "token_budget": (CouplingKind.PARAMETER, "configures a token budget"),
    "region": (CouplingKind.CONFIGURATION, "configures model region/routing"),
    "pricing": (CouplingKind.CONFIGURATION, "configures model pricing/cost telemetry"),
}


def _config_coupling_findings(
    catalog: dict[str, ConfigDocument],
    known_model_ids: set[str],
    usage_anchored: set[str],
) -> list[ApplicationFinding]:
    """Semantic couplings from anchored config documents (V1.5.0-b).

    Structured documents carry no parse locations, so findings anchor to line
    1 with the exact key path in metadata.
    """
    findings: list[ApplicationFinding] = []
    for relative in sorted(catalog):
        document = catalog[relative]
        couplings = find_config_couplings(
            document, known_model_ids, usage_anchored=relative in usage_anchored
        )
        for coupling in couplings:
            kind, detail = _CONFIG_COUPLING_DETAILS[coupling.value_kind]
            key = coupling.key_path[-1]
            provider = platform = None
            if (
                coupling.value_kind == "model_id"
                and isinstance(coupling.value, str)
                and "anthropic." in coupling.value
            ):
                provider, platform = "anthropic", "amazon-bedrock"
            value: str | bool | int | float = (
                coupling.value if coupling.value_kind == "model_id" else f"{key}={coupling.value}"
            )
            findings.append(
                ApplicationFinding(
                    kind=kind,
                    provider=provider,
                    platform=platform,
                    value=value,
                    detail=f"{detail} at {'.'.join(coupling.key_path)}",
                    location=SourceLocation(path=relative, line=1, column=0),
                    metadata={
                        "config": relative,
                        "key_path": ".".join(coupling.key_path),
                        "value_kind": coupling.value_kind,
                        "name": key,
                        "value": coupling.value,
                    },
                )
            )
    return findings


def scannable_files(path: Path) -> tuple[list[Path], list[Path]]:
    """The exact file set one scan covers: (python files, prompt-source candidates).

    Shared with the worklist-snapshot staleness key, so "the scanned
    application content" in the key means precisely the files a scan reads.
    """
    if path.is_file():
        return [path], []
    entries = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and not any(part in _IGNORED_DIRECTORIES for part in item.relative_to(path).parts)
    ]
    python_files = [item for item in entries if item.suffix == ".py"]
    candidate_files = [item for item in entries if item.suffix.casefold() in PROMPT_SOURCE_SUFFIXES]
    return python_files, candidate_files


def scan_application(
    root: Path | str,
    *,
    prompt_sources: Sequence[str] | None = None,
    known_model_ids: Sequence[str] | None = None,
) -> ApplicationAnalysis:
    path = Path(root)
    if not path.exists():
        raise ValueError(f"application path does not exist: {path}")
    base = path if path.is_dir() else path.parent
    python_files, candidate_files = scannable_files(path)
    candidate_relative = [item.relative_to(base).as_posix() for item in candidate_files]
    known_files = {
        *(item.relative_to(base).as_posix() for item in python_files),
        *candidate_relative,
    }
    catalog, config_warnings = load_config_documents(base, candidate_relative)
    findings: list[ApplicationFinding] = []
    tool_definitions: list[ToolDefinition] = []
    structured_outputs: list[StructuredOutputContract] = []
    multimodal_inputs: list[MultimodalInputContract] = []
    warnings: list[str] = []
    discovered: dict[str, list[str]] = {}
    config_usage: set[str] = set()
    for file_path in python_files:
        relative = file_path.relative_to(base).as_posix()
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            warnings.append(f"{relative}: {exc}")
            continue
        visitor = _Visitor(relative, catalog, known_files)
        visitor.visit(tree)
        findings.extend(visitor.findings)
        tool_definitions.extend(visitor.tool_definitions)
        structured_outputs.extend(visitor.structured_outputs)
        multimodal_inputs.extend(visitor.multimodal_inputs)
        warnings.extend(visitor.notes)
        config_usage.update(visitor.config_invocation_usage)
        for source_path, provenance in visitor.prompt_source_facts.items():
            discovered.setdefault(source_path, provenance)
    findings.extend(_config_coupling_findings(catalog, set(known_model_ids or ()), config_usage))
    warnings.extend(config_warnings)
    sources, source_warnings = assemble_prompt_sources(
        base, catalog, known_files, discovered, prompt_sources
    )
    warnings.extend(source_warnings)
    findings.sort(
        key=lambda item: (item.location.path, item.location.line, item.kind.value, item.detail)
    )
    return ApplicationAnalysis(
        root=str(path),
        files_scanned=len(python_files) + len(candidate_relative),
        findings=findings,
        requirements=_requirements(findings),
        prompt_sources=sources,
        prompt_discovery=summarize_prompt_discovery(findings, sources),
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
