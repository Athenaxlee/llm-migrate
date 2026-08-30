"""Credential-free contracts for V0.5 evaluation and regression artifacts."""

from __future__ import annotations

import json
from base64 import b64decode
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, field_serializer, field_validator, model_validator

from llm_migrate.core.models import MigrationEndpoint, StrictModel

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "bearer_token",
    "token",
    "client_secret",
    "secret",
    "password",
    "credential",
    "credentials",
    "authorization",
}
_RESERVED_REQUEST_KEYS = {"model", "input", "messages", "instructions", "system"}


def _looks_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or any(
        normalized.endswith(suffix)
        for suffix in ("_api_key", "_secret", "_password", "_credential")
    )


def _sensitive_parameter_path(value: Any, path: str = "parameters") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _looks_sensitive_key(str(key)):
                return child_path
            found = _sensitive_parameter_path(child, child_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _sensitive_parameter_path(child, f"{path}[{index}]")
            if found:
                return found
    return None


class EvaluatorKind(StrEnum):
    EXACT_MATCH = "exact_match"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    TOOL_CALLS = "tool_calls"
    CUSTOM = "custom"
    LLM_JUDGE = "llm_judge"


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class RegressionCategory(StrEnum):
    FORMAT_SCHEMA_PARSER = "format_schema_parser_incompatibility"
    HALLUCINATION_GROUNDING = "hallucination_or_grounding_regression"
    INSTRUCTION_ADHERENCE = "instruction_adherence_regression"
    REASONING = "over_or_under_reasoning"
    TOOL_USE = "tool_use_regression"
    REFUSAL_SAFETY = "refusal_or_safety_change"
    VERBOSITY = "verbosity_change"
    CONTEXT_LOSS = "context_loss_or_truncation"
    PARAMETER_CAPABILITY = "parameter_or_capability_mismatch"
    LATENCY_COST = "latency_or_token_cost_regression"
    UNCLEAR = "non_deterministic_or_unclear_regression"


class EvaluationValidator(StrictModel):
    kind: EvaluatorKind
    name: str = Field(min_length=1)
    pattern: str | None = None
    schema_definition: dict[str, Any] | None = None
    evaluator_name: str | None = None
    minimum_score: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_configuration(self) -> EvaluationValidator:
        if self.kind is EvaluatorKind.REGEX and self.pattern is None:
            raise ValueError("regex validators require pattern")
        if self.kind is EvaluatorKind.JSON_SCHEMA and self.schema_definition is None:
            raise ValueError("json_schema validators require schema_definition")
        if self.kind in {EvaluatorKind.CUSTOM, EvaluatorKind.LLM_JUDGE} and not self.evaluator_name:
            raise ValueError(f"{self.kind} validators require evaluator_name")
        return self


class EvaluationTextContent(StrictModel):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class EvaluationImageContent(StrictModel):
    type: Literal["image"] = "image"
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data: str = Field(min_length=1, description="Base64-encoded image bytes without a data URL.")

    @field_validator("data")
    @classmethod
    def valid_base64(cls, value: str) -> str:
        try:
            b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("media data must be valid base64") from exc
        return value


class EvaluationDocumentContent(StrictModel):
    type: Literal["document"] = "document"
    media_type: Literal["application/pdf", "text/plain"]
    data: str = Field(min_length=1, description="Base64-encoded document bytes.")
    title: str | None = None

    @field_validator("data")
    @classmethod
    def valid_base64(cls, value: str) -> str:
        try:
            b64decode(value, validate=True)
        except ValueError as exc:
            raise ValueError("media data must be valid base64") from exc
        return value

    @model_validator(mode="after")
    def plain_text_is_utf8(self) -> EvaluationDocumentContent:
        if self.media_type == "text/plain":
            try:
                b64decode(self.data, validate=True).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("text/plain document data must decode as UTF-8") from exc
        return self


EvaluationContent = Annotated[
    EvaluationTextContent | EvaluationImageContent | EvaluationDocumentContent,
    Field(discriminator="type"),
]


class EvaluationMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: list[EvaluationContent] = Field(min_length=1)


class EvaluationCase(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    input: str | None = None
    messages: list[EvaluationMessage] = Field(default_factory=list)
    system_prompt: str | None = None
    expected_output: Any | None = None
    expected_tool_calls: list[dict[str, Any]] | None = None
    validators: list[EvaluationValidator] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validator_inputs_exist(self) -> EvaluationCase:
        if self.input is None and not self.messages:
            raise ValueError("evaluation cases require input or messages")
        if self.input is not None and self.messages:
            raise ValueError("evaluation cases must use either input or messages, not both")
        if self.messages and self.messages[-1].role != "user":
            raise ValueError("the final evaluation message must use the user role")
        if any(
            block.type in {"image", "document"} and message.role != "user"
            for message in self.messages
            for block in message.content
        ):
            raise ValueError("image and document content is allowed only in user messages")
        kinds = {validator.kind for validator in self.validators}
        if EvaluatorKind.EXACT_MATCH in kinds and self.expected_output is None:
            raise ValueError("exact_match validators require expected_output")
        if EvaluatorKind.TOOL_CALLS in kinds and self.expected_tool_calls is None:
            raise ValueError("tool_calls validators require expected_tool_calls")
        return self


class EvaluationSuite(StrictModel):
    schema_version: Literal["1", "2"] = "2"
    name: str = Field(min_length=1)
    migration_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: MigrationEndpoint
    target: MigrationEndpoint
    cases: list[EvaluationCase] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> EvaluationSuite:
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("evaluation case ids must be unique")
        return self


class EvaluationRunConfig(StrictModel):
    provider: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    credential_env: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Name of a local environment variable; never the credential value.",
    )
    base_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=10)
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_price_per_million: Decimal | None = Field(default=None, ge=0)
    output_price_per_million: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def reject_inline_secrets(self) -> EvaluationRunConfig:
        reserved = _RESERVED_REQUEST_KEYS.intersection(key.lower() for key in self.parameters)
        if reserved:
            raise ValueError(
                "parameters must not override reserved request fields: "
                + ", ".join(sorted(reserved))
            )
        sensitive_path = _sensitive_parameter_path(self.parameters)
        if sensitive_path:
            raise ValueError(f"{sensitive_path} may contain a secret; use credential_env instead")
        try:
            json.dumps(self.parameters)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must contain only JSON-serializable values") from exc
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("base_url must be an absolute HTTP(S) URL")
            if parsed.username or parsed.password:
                raise ValueError("base_url must not contain user information")
            if parsed.fragment:
                raise ValueError("base_url must not contain a URL fragment")
            for key, _ in parse_qsl(parsed.query):
                if _looks_sensitive_key(key):
                    raise ValueError("base_url must not contain credential query parameters")
        return self

    @field_serializer("input_price_per_million", "output_price_per_million")
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class ProviderCallResult(StrictModel):
    status: EvaluationStatus
    output: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    refusal_detected: bool = False
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    attempts: int = Field(default=1, ge=1)
    error_kind: str | None = None
    error_message: str | None = None


class EvaluatorResult(StrictModel):
    name: str
    kind: EvaluatorKind
    status: EvaluationStatus
    score: float | None = Field(default=None, ge=0, le=1)
    message: str


class EvaluationCaseResult(StrictModel):
    case_id: str
    configuration: Literal["source", "target"]
    status: EvaluationStatus
    output: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    refusal_detected: bool = False
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
    task_score: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    attempts: int = Field(default=1, ge=1)
    error_kind: str | None = None
    error_message: str | None = None

    @field_serializer("estimated_cost_usd")
    def serialize_cost(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class MigrationEvalRun(StrictModel):
    schema_version: Literal["1"] = "1"
    suite: EvaluationSuite
    source_config: EvaluationRunConfig
    target_config: EvaluationRunConfig
    started_at: datetime
    completed_at: datetime
    source_results: list[EvaluationCaseResult]
    target_results: list[EvaluationCaseResult]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def results_match_suite(self) -> MigrationEvalRun:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        case_ids = [case.id for case in self.suite.cases]
        if [result.case_id for result in self.source_results] != case_ids:
            raise ValueError("source results must match suite case order")
        if [result.case_id for result in self.target_results] != case_ids:
            raise ValueError("target results must match suite case order")
        if any(result.configuration != "source" for result in self.source_results):
            raise ValueError("source results must use the source configuration label")
        if any(result.configuration != "target" for result in self.target_results):
            raise ValueError("target results must use the target configuration label")
        source_endpoint = (
            self.source_config.model,
            self.source_config.provider,
            self.source_config.platform,
            self.source_config.model_id,
        )
        target_endpoint = (
            self.target_config.model,
            self.target_config.provider,
            self.target_config.platform,
            self.target_config.model_id,
        )
        if source_endpoint != (
            self.suite.source.model,
            self.suite.source.provider,
            self.suite.source.platform,
            self.suite.source.model_id,
        ):
            raise ValueError("source config must match the suite endpoint")
        if target_endpoint != (
            self.suite.target.model,
            self.suite.target.provider,
            self.suite.target.platform,
            self.suite.target.model_id,
        ):
            raise ValueError("target config must match the suite endpoint")
        return self


class OutputComparison(StrictModel):
    schema_version: Literal["1"] = "1"
    case_id: str
    source_status: EvaluationStatus
    target_status: EvaluationStatus
    source_score: float | None = None
    target_score: float | None = None
    score_delta: float | None = None
    outputs_equal: bool | None = None
    source_latency_ms: float
    target_latency_ms: float
    latency_delta_ms: float
    source_tokens: int | None = None
    target_tokens: int | None = None
    token_delta: int | None = None
    source_cost_usd: Decimal | None = None
    target_cost_usd: Decimal | None = None
    cost_delta_usd: Decimal | None = None
    regression: bool
    signals: list[str] = Field(default_factory=list)

    @field_serializer("source_cost_usd", "target_cost_usd", "cost_delta_usd")
    def serialize_costs(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class RegressionFinding(StrictModel):
    case_id: str
    category: RegressionCategory
    severity: Literal["warning", "regression", "blocker"]
    summary: str
    evidence: list[str] = Field(default_factory=list)


class RegressionSummary(StrictModel):
    case_count: int = Field(ge=0)
    source_pass_rate: float = Field(ge=0, le=1)
    target_pass_rate: float = Field(ge=0, le=1)
    pass_rate_delta: float
    regression_count: int = Field(ge=0)
    source_error_count: int = Field(ge=0)
    target_error_count: int = Field(ge=0)
    source_refusal_rate: float = Field(ge=0, le=1)
    target_refusal_rate: float = Field(ge=0, le=1)


class RegressionReport(StrictModel):
    schema_version: Literal["1", "2"] = "2"
    suite_name: str
    evaluation_suite_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    migration_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source: MigrationEndpoint
    target: MigrationEndpoint
    summary: RegressionSummary
    comparisons: list[OutputComparison]
    regressions: list[RegressionFinding]
    warnings: list[str] = Field(default_factory=list)


class OptimizationLimits(StrictModel):
    objective: Literal["balanced", "quality", "cost", "latency"] = "balanced"
    max_candidates: int = Field(default=3, ge=1, le=10)
    max_evaluation_runs: int = Field(default=3, ge=0, le=20)
    max_estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    minimum_target_pass_rate: float = Field(default=1.0, ge=0, le=1)
    allow_quality_regression: bool = False

    @field_serializer("max_estimated_cost_usd")
    def serialize_cost(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class OptimizationRecommendation(StrictModel):
    category: RegressionCategory
    action: Literal["prompt", "parameter", "model", "tool_schema", "output_contract"]
    summary: str
    rationale: str
    affected_cases: list[str] = Field(default_factory=list)
    review_required: bool = True


class OptimizationCandidateComparison(StrictModel):
    candidate_id: str
    config: EvaluationRunConfig
    target_pass_rate: float = Field(ge=0, le=1)
    pass_rate_delta_from_baseline: float
    regression_count: int = Field(ge=0)
    mean_target_latency_ms: float = Field(ge=0)
    estimated_target_cost_usd: Decimal | None = Field(default=None, ge=0)
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)

    @field_serializer("estimated_target_cost_usd")
    def serialize_cost(self, value: Decimal | None) -> str | None:
        return str(value) if value is not None else None


class OptimizationResult(StrictModel):
    schema_version: Literal["1"] = "1"
    algorithm_version: Literal["bounded-v1"] = "bounded-v1"
    regression_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reproducibility_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limits: OptimizationLimits
    recommendations: list[OptimizationRecommendation]
    candidates: list[OptimizationCandidateComparison]
    selected_candidate_id: str | None = None
    evaluated_run_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
