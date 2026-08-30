"""Local BYOK runtime adapters for source/target evaluation."""

from __future__ import annotations

import json
import os
import re
import time
from base64 import b64decode
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationStatus,
    ProviderCallResult,
)


def _openai_input(case: EvaluationCase) -> str | list[dict[str, Any]]:
    if case.input is not None:
        return case.input
    messages: list[dict[str, Any]] = []
    for message in case.messages:
        content: list[dict[str, Any]] = []
        for block in message.content:
            if block.type == "text":
                content.append({"type": "input_text", "text": block.text})
            elif block.type == "image":
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{block.media_type};base64,{block.data}",
                    }
                )
            else:
                content.append(
                    {
                        "type": "input_file",
                        "filename": block.title or "document",
                        "file_data": f"data:{block.media_type};base64,{block.data}",
                    }
                )
        messages.append({"role": message.role, "content": content})
    return messages


def _anthropic_messages(case: EvaluationCase) -> list[dict[str, Any]]:
    if case.input is not None:
        return [{"role": "user", "content": case.input}]
    messages: list[dict[str, Any]] = []
    for message in case.messages:
        content: list[dict[str, Any]] = []
        for block in message.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "image":
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.media_type,
                            "data": block.data,
                        },
                    }
                )
            else:
                source = (
                    {
                        "type": "text",
                        "media_type": block.media_type,
                        "data": b64decode(block.data).decode("utf-8"),
                    }
                    if block.media_type == "text/plain"
                    else {"type": "base64", "media_type": block.media_type, "data": block.data}
                )
                content.append(
                    {
                        "type": "document",
                        "source": source,
                        **({"title": block.title} if block.title else {}),
                    }
                )
        messages.append({"role": message.role, "content": content})
    return messages


class EvaluationTransport(Protocol):
    def __call__(self, request: Request, timeout: float) -> tuple[int, dict[str, Any]]: ...


def _urlopen_transport(request: Request, timeout: float) -> tuple[int, dict[str, Any]]:
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit BYOK endpoint
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        status = exc.code
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
    if not isinstance(payload, dict):
        raise ValueError("provider response root must be an object")
    return status, payload


class HttpEvaluationExecutor:
    """Minimal Anthropic/OpenAI direct-API executor with bounded retries."""

    def __init__(
        self,
        *,
        transport: EvaluationTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.transport = transport or _urlopen_transport
        self.sleeper = sleeper
        self.monotonic = monotonic

    def _request(
        self, config: EvaluationRunConfig, case: EvaluationCase, credential: str
    ) -> Request:
        provider = config.provider.lower()
        if provider == "openai":
            url = (config.base_url or "https://api.openai.com").rstrip("/") + "/v1/responses"
            payload: dict[str, Any] = {
                **config.parameters,
                "model": config.model_id,
                "input": _openai_input(case),
            }
            if case.system_prompt:
                payload["instructions"] = case.system_prompt
            headers = {"Authorization": f"Bearer {credential}"}
        elif provider == "anthropic":
            url = (config.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
            payload = {
                "max_tokens": 1024,
                **config.parameters,
                "model": config.model_id,
                "messages": _anthropic_messages(case),
            }
            if case.system_prompt:
                payload["system"] = case.system_prompt
            headers = {"x-api-key": credential, "anthropic-version": "2023-06-01"}
        else:
            raise ValueError(f"unsupported evaluation provider {config.provider!r}")
        headers["Content-Type"] = "application/json"
        return Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

    @staticmethod
    def _parse_openai(
        payload: dict[str, Any],
    ) -> tuple[str | None, list[dict[str, Any]], bool]:
        texts: list[str] = []
        tools: list[dict[str, Any]] = []
        refusal = False
        output = payload.get("output", [])
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"function_call", "tool_call"}:
                    arguments = item.get("arguments", {})
                    if isinstance(arguments, str):
                        with suppress(json.JSONDecodeError):
                            arguments = json.loads(arguments)
                    tools.append({"name": item.get("name"), "arguments": arguments})
                content = item.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            texts.append(block["text"])
                        elif isinstance(block, dict) and block.get("type") == "refusal":
                            refusal = True
                            if isinstance(block.get("refusal"), str):
                                texts.append(block["refusal"])
        if not texts and isinstance(payload.get("output_text"), str):
            texts.append(payload["output_text"])
        return "".join(texts) if texts else None, tools, refusal

    @staticmethod
    def _parse_anthropic(
        payload: dict[str, Any],
    ) -> tuple[str | None, list[dict[str, Any]], bool]:
        texts: list[str] = []
        tools: list[dict[str, Any]] = []
        content = payload.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tools.append({"name": block.get("name"), "arguments": block.get("input", {})})
        return "".join(texts) if texts else None, tools, payload.get("stop_reason") == "refusal"

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        started = self.monotonic()
        supported_endpoints = {("anthropic", "anthropic-api"), ("openai", "openai-api")}
        if (config.provider.lower(), config.platform.lower()) not in supported_endpoints:
            return ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=0,
                error_kind="unsupported",
                error_message=(
                    "The built-in HTTP evaluator supports only anthropic/anthropic-api "
                    "and openai/openai-api; inject an EvaluationExecutor for this endpoint."
                ),
            )
        if not config.credential_env:
            return ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=0,
                error_kind="authentication",
                error_message="credential_env is required for runtime provider evaluation.",
            )
        credential = os.environ.get(config.credential_env)
        if not credential:
            return ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=0,
                error_kind="authentication",
                error_message=(
                    f"Credential environment variable {config.credential_env!r} is not set."
                ),
            )
        try:
            request = self._request(config, case, credential)
        except ValueError as exc:
            return ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=0,
                error_kind="unsupported",
                error_message=str(exc),
            )
        attempts = 0
        while attempts <= config.max_retries:
            attempts += 1
            try:
                status, payload = self.transport(request, config.timeout_seconds)
            except (TimeoutError, URLError, OSError, ValueError):
                if attempts <= config.max_retries:
                    self.sleeper(min(2 ** (attempts - 1), 8))
                    continue
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    attempts=attempts,
                    error_kind="network",
                    error_message="Provider request failed after bounded retries.",
                )
            if status == 429 or status >= 500:
                if attempts <= config.max_retries:
                    self.sleeper(min(2 ** (attempts - 1), 8))
                    continue
                error_kind = "rate_limit" if status == 429 else "provider_unavailable"
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    attempts=attempts,
                    error_kind=error_kind,
                    error_message="Provider request failed after bounded retries.",
                )
            if status >= 400:
                error_kind = "authentication" if status in {401, 403} else "invalid_request"
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    attempts=attempts,
                    error_kind=error_kind,
                    error_message=f"Provider returned HTTP {status}.",
                )
            if config.provider.lower() == "openai":
                output, tool_calls, refusal = self._parse_openai(payload)
            else:
                output, tool_calls, refusal = self._parse_anthropic(payload)
            usage = payload.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if not isinstance(input_tokens, int):
                input_tokens = None
            if not isinstance(output_tokens, int):
                output_tokens = None
            if output is None and not tool_calls:
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    attempts=attempts,
                    error_kind="invalid_output",
                    error_message="Provider response contained neither text nor tool calls.",
                )
            return ProviderCallResult(
                status=EvaluationStatus.PASSED,
                output=output or "",
                tool_calls=tool_calls,
                refusal_detected=refusal,
                latency_ms=(self.monotonic() - started) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                attempts=attempts,
            )
        raise AssertionError("unreachable retry loop")


class BedrockRuntimeClient(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


class _BedrockDependencyError(RuntimeError):
    pass


def _bedrock_messages(case: EvaluationCase) -> list[dict[str, Any]]:
    if case.input is not None:
        return [{"role": "user", "content": [{"text": case.input}]}]
    messages: list[dict[str, Any]] = []
    image_count = 0
    document_count = 0
    for message in case.messages:
        content: list[dict[str, Any]] = []
        has_text = any(block.type == "text" for block in message.content)
        for index, block in enumerate(message.content):
            if block.type == "text":
                content.append({"text": block.text})
            elif block.type == "image":
                image_count += 1
                image_bytes = b64decode(block.data, validate=True)
                if len(image_bytes) > 3_750_000:
                    raise ValueError("Bedrock image inputs must not exceed 3.75 MB")
                image_format = {
                    "image/jpeg": "jpeg",
                    "image/png": "png",
                    "image/gif": "gif",
                    "image/webp": "webp",
                }[block.media_type]
                content.append(
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        }
                    }
                )
            else:
                document_count += 1
                if not has_text:
                    raise ValueError("Bedrock document messages require an accompanying text block")
                document_bytes = b64decode(block.data, validate=True)
                if len(document_bytes) > 4_500_000:
                    raise ValueError("Bedrock document inputs must not exceed 4.5 MB")
                document_format = "pdf" if block.media_type == "application/pdf" else "txt"
                name = re.sub(r"[^A-Za-z0-9 ()\[\]-]+", "-", block.title or f"document-{index + 1}")
                content.append(
                    {
                        "document": {
                            "format": document_format,
                            "name": name[:200],
                            "source": {"bytes": document_bytes},
                        }
                    }
                )
        messages.append({"role": message.role, "content": content})
    if image_count > 20:
        raise ValueError("Bedrock evaluation cases support at most 20 images")
    if document_count > 5:
        raise ValueError("Bedrock evaluation cases support at most 5 documents")
    return messages


class BedrockEvaluationExecutor:
    """AWS Bedrock Converse executor using the caller's normal local AWS credential chain."""

    def __init__(
        self,
        client: BedrockRuntimeClient | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.sleeper = sleeper
        self.monotonic = monotonic

    def _client(self, config: EvaluationRunConfig) -> BedrockRuntimeClient:
        if self.client is not None:
            return self.client
        try:
            import boto3  # type: ignore[import-not-found]
            from botocore.config import Config  # type: ignore[import-not-found]
        except ImportError as exc:
            raise _BedrockDependencyError from exc
        region = config.parameters.get("region_name")
        sdk_config = Config(
            connect_timeout=config.timeout_seconds,
            read_timeout=config.timeout_seconds,
            retries={"total_max_attempts": 1, "mode": "standard"},
        )
        return cast(
            BedrockRuntimeClient,
            boto3.client(
                "bedrock-runtime",
                config=sdk_config,
                **({"region_name": region} if region else {}),
            ),
        )

    @staticmethod
    def _error_details(exc: Exception) -> tuple[bool, str]:
        response = getattr(exc, "response", None)
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        code = error.get("Code") if isinstance(error, dict) else None
        identifier = code if isinstance(code, str) else type(exc).__name__
        retryable = identifier in {
            "ThrottlingException",
            "ServiceUnavailableException",
            "ModelTimeoutException",
            "InternalServerException",
            "ModelNotReadyException",
            "ReadTimeoutError",
            "EndpointConnectionError",
            "ConnectionClosedError",
        }
        if identifier == "ThrottlingException":
            return retryable, "rate_limit"
        if retryable:
            return retryable, "provider_unavailable"
        if identifier in {"AccessDeniedException", "UnrecognizedClientException"}:
            return False, "authentication"
        if identifier in {
            "NoCredentialsError",
            "PartialCredentialsError",
            "CredentialRetrievalError",
            "ProfileNotFound",
        }:
            return False, "authentication"
        if identifier in {"ValidationException", "ResourceNotFoundException"}:
            return False, "invalid_request"
        return False, "provider_error"

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        if (config.provider.lower(), config.platform.lower()) != (
            "anthropic",
            "amazon-bedrock",
        ):
            return ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=0,
                error_kind="unsupported",
                error_message=(
                    "The built-in Bedrock evaluator supports only "
                    "anthropic/amazon-bedrock endpoints."
                ),
            )
        started = self.monotonic()
        parameters = dict(config.parameters)
        parameters.pop("region_name", None)
        try:
            messages = _bedrock_messages(case)
            request: dict[str, Any] = {"modelId": config.model_id, "messages": messages}
            if case.system_prompt:
                request["system"] = [{"text": case.system_prompt}]
            inference_keys = {"maxTokens", "stopSequences", "temperature", "topP"}
            inference_config = {
                key: parameters.pop(key) for key in list(parameters) if key in inference_keys
            }
            if inference_config:
                request["inferenceConfig"] = inference_config
            top_level_keys = {
                "toolConfig",
                "guardrailConfig",
                "additionalModelResponseFieldPaths",
                "performanceConfig",
                "requestMetadata",
                "outputConfig",
                "serviceTier",
            }
            for key in list(parameters):
                if key in top_level_keys:
                    request[key] = parameters.pop(key)
            if parameters:
                request["additionalModelRequestFields"] = parameters
        except (ValueError, TypeError):
            return ProviderCallResult(
                status=EvaluationStatus.ERROR,
                latency_ms=0,
                error_kind="invalid_input",
                error_message="Evaluation media is invalid for Amazon Bedrock.",
            )
        attempts = 0
        while attempts <= config.max_retries:
            attempts += 1
            try:
                payload = self._client(config).converse(**request)
            except _BedrockDependencyError:
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    attempts=attempts,
                    error_kind="unsupported",
                    error_message=("Bedrock evaluation requires the optional 'aws' package extra."),
                )
            except Exception as exc:  # SDK exceptions vary; never serialize their message.
                retryable, error_kind = self._error_details(exc)
                if retryable and attempts <= config.max_retries:
                    self.sleeper(min(2 ** (attempts - 1), 8))
                    continue
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    attempts=attempts,
                    error_kind=error_kind,
                    error_message=f"Bedrock request failed with {type(exc).__name__}.",
                )
            if not isinstance(payload, dict):
                return ProviderCallResult(
                    status=EvaluationStatus.ERROR,
                    latency_ms=(self.monotonic() - started) * 1000,
                    attempts=attempts,
                    error_kind="invalid_output",
                    error_message="Bedrock response root was not an object.",
                )
            output_container = payload.get("output", {})
            output_message = (
                output_container.get("message", {}) if isinstance(output_container, dict) else {}
            )
            content = output_message.get("content", []) if isinstance(output_message, dict) else []
            texts: list[str] = []
            tools: list[dict[str, Any]] = []
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                if isinstance(block, dict) and isinstance(block.get("toolUse"), dict):
                    tool = block["toolUse"]
                    tools.append({"name": tool.get("name"), "arguments": tool.get("input", {})})
            usage = payload.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}
            return ProviderCallResult(
                status=EvaluationStatus.PASSED if texts or tools else EvaluationStatus.ERROR,
                output="".join(texts) if texts else "",
                tool_calls=tools,
                refusal_detected=payload.get("stopReason")
                in {"guardrail_intervened", "content_filtered"},
                latency_ms=(self.monotonic() - started) * 1000,
                input_tokens=usage.get("inputTokens")
                if isinstance(usage.get("inputTokens"), int)
                else None,
                output_tokens=usage.get("outputTokens")
                if isinstance(usage.get("outputTokens"), int)
                else None,
                attempts=attempts,
                error_kind=None if texts or tools else "invalid_output",
                error_message=None
                if texts or tools
                else "Bedrock response contained neither text nor tool calls.",
            )
        raise AssertionError("unreachable retry loop")


class BuiltinEvaluationExecutor:
    """Route only reviewed built-in platform adapters; every other platform fails closed."""

    def __init__(
        self,
        http: HttpEvaluationExecutor | None = None,
        bedrock: BedrockEvaluationExecutor | None = None,
    ) -> None:
        self.http = http or HttpEvaluationExecutor()
        self.bedrock = bedrock or BedrockEvaluationExecutor()

    def execute(self, config: EvaluationRunConfig, case: EvaluationCase) -> ProviderCallResult:
        if config.platform.lower() == "amazon-bedrock":
            return self.bedrock.execute(config, case)
        return self.http.execute(config, case)
