"""Validated domain models used across the registry, CLI, and MCP server."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_serializer, model_validator


class StrictModel(BaseModel):
    """Base class that rejects misspelled or unknown registry fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceType(StrEnum):
    OFFICIAL_STRUCTURED = "official_structured"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    PROVIDER_ANNOUNCEMENT = "provider_announcement"
    OFFICIAL_REPOSITORY = "official_repository"
    INDEPENDENT_EVIDENCE = "independent_evidence"
    COMMUNITY_REPORT = "community_report"
    TEST_FIXTURE = "test_fixture"


class AuthorityTier(IntEnum):
    AUTHORITATIVE_STRUCTURED = 1
    OFFICIAL_DOCUMENTATION = 2
    OFFICIAL_ANNOUNCEMENT = 3
    OFFICIAL_REPOSITORY = 4
    CREDIBLE_INDEPENDENT = 5
    COMMUNITY = 6


class SourceReference(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    url: HttpUrl | None = None
    title: str
    source_type: SourceType
    authority_tier: AuthorityTier
    publisher: str
    retrieved_at: date
    published_at: date | None = None
    supports: list[str] = Field(default_factory=list)
    notes: str | None = None


class ModelIdentity(StrictModel):
    canonical_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    version: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)


class CapabilityOverrides(StrictModel):
    """Explicit platform differences; omitted values inherit the model profile."""

    text_input: bool | None = None
    image_input: bool | None = None
    document_input: bool | None = None
    structured_output: bool | None = None
    tool_use: bool | None = None
    parallel_tool_use: bool | None = None
    reasoning: bool | None = None
    prompt_caching: bool | None = None
    streaming: bool | None = None
    batch_inference: bool | None = None
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)


class PlatformAvailability(StrictModel):
    platform: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    endpoint: str | None = None
    regions: list[str] = Field(default_factory=list)
    capability_overrides: CapabilityOverrides | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class LifecycleStatus(StrEnum):
    ACTIVE = "active"
    LEGACY = "legacy"
    DEPRECATED = "deprecated"
    END_OF_LIFE = "end_of_life"


class ModelLifecycle(StrictModel):
    status: LifecycleStatus
    announced_on: date | None = None
    deprecated_on: date | None = None
    end_of_life_on: date | None = None
    earliest_end_of_life_on: date | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class PricePerMillionTokens(StrictModel):
    amount: Decimal = Field(ge=0)
    currency: Literal["USD"] = "USD"
    unit: Literal["per_1m_tokens"] = "per_1m_tokens"
    valid_from: date | None = None
    valid_until: date | None = None
    notes: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class ModelPricing(StrictModel):
    input: PricePerMillionTokens | None = None
    output: PricePerMillionTokens | None = None
    cached_input: PricePerMillionTokens | None = None
    batch: PricePerMillionTokens | None = None


class ModelLifecycleCheck(StrictModel):
    """Point-in-time lifecycle interpretation without mutating registry facts."""

    schema_version: Literal["1"] = "1"
    model: str
    status: LifecycleStatus
    as_of_date: date
    suitable_for_new_migrations: bool
    days_until_end_of_life: int | None = None
    lifecycle_facts_stale: bool
    warnings: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)


class MigrationWorkload(StrictModel):
    """Normalized workload used for a deterministic recurring-cost comparison."""

    requests: int = Field(gt=0)
    input_tokens_per_request: int = Field(ge=0)
    output_tokens_per_request: int = Field(ge=0)


class EndpointCostEstimate(StrictModel):
    model: str
    input_cost_usd: Decimal | None = Field(default=None, ge=0)
    output_cost_usd: Decimal | None = Field(default=None, ge=0)
    total_cost_usd: Decimal | None = Field(default=None, ge=0)
    unknown_price_components: list[Literal["input", "output"]] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)


class MigrationCostEstimate(StrictModel):
    """Canonical-price estimate for the same workload on source and target models."""

    schema_version: Literal["1"] = "1"
    workload: MigrationWorkload
    source: EndpointCostEstimate
    target: EndpointCostEstimate
    estimated_delta_usd: Decimal | None = None
    estimated_savings_percent: Decimal | None = None
    caveats: list[str] = Field(default_factory=list)


class ModelCapabilities(StrictModel):
    text_input: bool | None = None
    image_input: bool | None = None
    document_input: bool | None = None
    structured_output: bool | None = None
    tool_use: bool | None = None
    parallel_tool_use: bool | None = None
    reasoning: bool | None = None
    prompt_caching: bool | None = None
    streaming: bool | None = None
    batch_inference: bool | None = None
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    sources: list[SourceReference] = Field(default_factory=list)


class ParameterState(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    DEPRECATED = "deprecated"
    REQUIRED = "required"


class ParameterSupport(StrictModel):
    state: ParameterState
    default: Any | None = None
    notes: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class PromptGuidance(StrictModel):
    preferred_structure: list[str] = Field(default_factory=list)
    delimiter_guidance: list[str] = Field(default_factory=list)
    reasoning_guidance: list[str] = Field(default_factory=list)
    verbosity_sensitivity: list[str] = Field(default_factory=list)
    negative_instruction_guidance: list[str] = Field(default_factory=list)
    structured_output_guidance: list[str] = Field(default_factory=list)
    tool_use_guidance: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)


class FreshnessCategory(StrEnum):
    PRICING = "pricing"
    LIFECYCLE = "lifecycle"
    AVAILABILITY = "availability"
    CAPABILITIES = "capabilities"
    PROMPTING_GUIDANCE = "prompting_guidance"


class FreshnessMetadata(StrictModel):
    checked_at: date
    max_age_days: int = Field(gt=0)

    def is_stale(self, as_of_date: date) -> bool:
        return as_of_date > self.checked_at + timedelta(days=self.max_age_days)


class RegistryEvidenceConflict(StrictModel):
    field_path: str
    statements: list[str] = Field(min_length=2)
    source_ids: list[str] = Field(min_length=2)
    notes: str | None = None


class ModelProfile(StrictModel):
    schema_version: Literal["1"] = "1"
    fixture: bool = False
    identity: ModelIdentity
    platforms: list[PlatformAvailability] = Field(min_length=1)
    lifecycle: ModelLifecycle
    pricing: ModelPricing | None = None
    capabilities: ModelCapabilities
    parameters: dict[str, ParameterSupport] = Field(default_factory=dict)
    prompt_guidance: PromptGuidance = Field(default_factory=PromptGuidance)
    freshness: dict[FreshnessCategory, FreshnessMetadata] = Field(default_factory=dict)
    evidence_conflicts: list[RegistryEvidenceConflict] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_provenance(self) -> ModelProfile:
        if not self.fixture and not self.sources:
            raise ValueError("non-fixture profiles require at least one top-level source")
        source_ids = {source.id for source in self.sources}
        for conflict in self.evidence_conflicts:
            missing = set(conflict.source_ids) - source_ids
            if missing:
                raise ValueError(f"unknown model conflict source references: {sorted(missing)}")
        return self


class IdentifierMatchType(StrEnum):
    CANONICAL_NAME = "canonical_name"
    DISPLAY_NAME = "display_name"
    ALIAS = "alias"
    PLATFORM_MODEL_ID = "platform_model_id"


class ResolvedModel(StrictModel):
    """A canonical model plus the platform context implied or requested by the caller."""

    profile: ModelProfile
    matched_identifier: str
    matched_by: IdentifierMatchType
    platform: PlatformAvailability | None = None
    effective_capabilities: ModelCapabilities

    @property
    def canonical_name(self) -> str:
        return self.profile.identity.canonical_name

    @property
    def identity(self) -> ModelIdentity:
        """Compatibility accessor for callers that previously received a profile."""
        return self.profile.identity

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.effective_capabilities

    @property
    def platforms(self) -> list[PlatformAvailability]:
        return self.profile.platforms

    @property
    def lifecycle(self) -> ModelLifecycle:
        return self.profile.lifecycle

    @property
    def pricing(self) -> ModelPricing | None:
        return self.profile.pricing

    @property
    def parameters(self) -> dict[str, ParameterSupport]:
        return self.profile.parameters

    @property
    def prompt_guidance(self) -> PromptGuidance:
        return self.profile.prompt_guidance

    @property
    def freshness(self) -> dict[FreshnessCategory, FreshnessMetadata]:
        return self.profile.freshness

    @property
    def evidence_conflicts(self) -> list[RegistryEvidenceConflict]:
        return self.profile.evidence_conflicts

    @property
    def fixture(self) -> bool:
        return self.profile.fixture

    @property
    def sources(self) -> list[SourceReference]:
        return self.profile.sources


class PricingMatchStatus(StrEnum):
    MATCHED = "matched"
    UNMAPPED = "unmapped"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


class LivePricingResult(StrictModel):
    canonical_name: str
    openrouter_model_id: str | None = None
    source_url: str = "https://openrouter.ai/api/v1/models"
    retrieved_at: datetime
    status: PricingMatchStatus
    scope: Literal["openrouter"] = "openrouter"
    raw_prompt_per_token: str | None = None
    raw_completion_per_token: str | None = None
    input_per_million: Decimal | None = None
    output_per_million: Decimal | None = None
    registry_input_per_million: Decimal | None = None
    registry_output_per_million: Decimal | None = None
    discrepancies: list[str] = Field(default_factory=list)
    error: str | None = None


class ComparisonSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BREAKING = "breaking"


class ComparisonState(StrEnum):
    SAME = "same"
    DIFFERENT = "different"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ModelDifference(StrictModel):
    category: str
    field: str | None = None
    source_value: Any
    target_value: Any
    state: ComparisonState = ComparisonState.DIFFERENT
    severity: ComparisonSeverity
    migration_impact: str
    recommended_action: str | None = None
    supporting_sources: list[str] = Field(default_factory=list)
    knowledge_id: str | None = None


class ModelComparison(StrictModel):
    source_model: str
    target_model: str
    source_platform: str | None = None
    target_platform: str | None = None
    differences: list[ModelDifference]
    highest_severity: ComparisonSeverity
    pricing_overlays: list[LivePricingResult] = Field(default_factory=list)


class MigrationGoal(StrEnum):
    UPGRADE = "upgrade"
    LOWER_COST = "lower_cost"
    HIGHER_QUALITY = "higher_quality"
    LOWER_LATENCY = "lower_latency"
    CROSS_PROVIDER = "cross_provider"
    BALANCED = "balanced"


class RecommendationConstraints(StrictModel):
    platform: str | None = None
    provider: str | None = None
    region: str | None = None
    required_capabilities: set[str] = Field(default_factory=set)
    minimum_context_window: int | None = Field(default=None, gt=0)
    migration_goal: MigrationGoal = MigrationGoal.BALANCED
    source_model: str | None = None
    source_platform: str | None = None

    @field_serializer("required_capabilities")
    def serialize_required_capabilities(self, value: set[str]) -> list[str]:
        return sorted(value)


class ApplicationRequirements(StrictModel):
    required_capabilities: set[str] = Field(default_factory=set)
    required_parameters: set[str] = Field(default_factory=set)
    minimum_context_window: int | None = Field(default=None, gt=0)
    source_models: set[str] = Field(default_factory=set)
    source_providers: set[str] = Field(default_factory=set)
    source_platforms: set[str] = Field(default_factory=set)

    @field_serializer(
        "required_capabilities",
        "required_parameters",
        "source_models",
        "source_providers",
        "source_platforms",
    )
    def serialize_sets(self, value: set[str]) -> list[str]:
        return sorted(value)


class ModelRecommendation(StrictModel):
    canonical_name: str
    platform: str | None = None
    score: Decimal
    reasons: list[str]
    blockers: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    stale_categories: list[FreshnessCategory] = Field(default_factory=list)
    evidence_state: Literal["verified", "conflicting", "fixture"] = "verified"


class ExcludedCandidate(StrictModel):
    canonical_name: str
    blockers: list[str]
    unknowns: list[str] = Field(default_factory=list)


class RecommendationResult(StrictModel):
    constraints: RecommendationConstraints
    application_requirements: ApplicationRequirements | None = None
    recommendations: list[ModelRecommendation]
    excluded_candidates: list[ExcludedCandidate] = Field(default_factory=list)
    pricing_overlays: list[LivePricingResult] = Field(default_factory=list)


class SourceLocation(StrictModel):
    path: str
    line: int = Field(ge=1)
    column: int = Field(ge=0)
    end_line: int | None = Field(default=None, ge=1)


class CouplingKind(StrEnum):
    PROVIDER_SDK = "provider_sdk"
    MODEL_IDENTIFIER = "model_identifier"
    INVOCATION = "invocation"
    PROMPT = "prompt"
    PARAMETER = "parameter"
    TOOL = "tool"
    STRUCTURED_OUTPUT = "structured_output"
    RESPONSE_PARSER = "response_parser"
    STREAMING = "streaming"
    MULTIMODAL = "multimodal"
    CONFIGURATION = "configuration"
    RETRY_ERROR_HANDLING = "retry_error_handling"


class ApplicationFinding(StrictModel):
    kind: CouplingKind
    provider: str | None = None
    platform: str | None = None
    value: str | bool | int | float | None = None
    detail: str
    location: SourceLocation
    metadata: dict[str, Any] | None = None


class ToolDefinition(StrictModel):
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    strict: bool | None = None
    provider_format: str
    choice: str | None = None
    location: SourceLocation


class StructuredOutputContract(StrictModel):
    name: str | None = None
    json_schema: dict[str, Any] | None = None
    strict: bool | None = None
    provider_format: str
    location: SourceLocation


class MultimodalInputContract(StrictModel):
    modality: Literal["image_input", "document_input"]
    provider_format: str
    source_kind: Literal["url", "base64", "bytes", "file", "unknown"]
    location: SourceLocation


class ApplicationAnalysis(StrictModel):
    schema_version: Literal["2"] = "2"
    root: str
    files_scanned: int
    findings: list[ApplicationFinding]
    requirements: ApplicationRequirements
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)
    structured_outputs: list[StructuredOutputContract] = Field(default_factory=list)
    multimodal_inputs: list[MultimodalInputContract] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"


class PromptFinding(StrictModel):
    category: str
    severity: FindingSeverity
    message: str
    evidence: list[str] = Field(default_factory=list)


class PromptIntent(StrictModel):
    """Provider-neutral, conservative decomposition of a prompt."""

    objective: list[str] = Field(default_factory=list)
    input_output_contracts: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    grounding_policies: list[str] = Field(default_factory=list)
    reasoning_policies: list[str] = Field(default_factory=list)
    tool_policies: list[str] = Field(default_factory=list)
    verbosity_policies: list[str] = Field(default_factory=list)


class PromptAnalysis(StrictModel):
    schema_version: Literal["2"] = "2"
    character_count: int
    word_count: int
    approximate_token_count: int
    markdown_headers: list[str]
    xml_like_tags: list[str]
    findings: list[PromptFinding]
    intent: PromptIntent = Field(default_factory=PromptIntent)
    caveat: str = "Static heuristics describe prompt characteristics; they do not measure quality."


class AdviceBasis(StrEnum):
    DETERMINISTIC = "deterministic_fact"
    HEURISTIC = "heuristic_recommendation"


class MigrationAdvice(StrictModel):
    text: str
    basis: AdviceBasis


class SemanticDiffState(StrEnum):
    PRESERVED = "preserved"
    REMOVED = "removed"
    REPLACED = "replaced"
    STRENGTHENED = "strengthened"
    SIMPLIFIED = "simplified"
    UNRESOLVED = "unresolved"


class PromptSemanticChange(StrictModel):
    state: SemanticDiffState
    concern: str
    before: str | None = None
    after: str | None = None
    rationale: str
    basis: AdviceBasis


class PromptMigrationSpec(StrictModel):
    schema_version: Literal["3"] = "3"
    source_model: str
    target_model: str
    source_provider: str
    target_provider: str
    source_platform: str | None = None
    target_platform: str | None = None
    source_prompt_sha256: str
    source_path: str | None = None
    source_role: Literal["system", "user", "developer", "unknown"] = "unknown"
    source_prompt_analysis: PromptAnalysis
    candidate_prompt: str
    semantic_diff: list[PromptSemanticChange]
    requirements_to_preserve: list[MigrationAdvice]
    assumptions_to_reconsider: list[MigrationAdvice]
    instructions_potentially_removable: list[MigrationAdvice]
    instructions_needing_strengthening: list[MigrationAdvice]
    target_features_replacing_prompt_text: list[MigrationAdvice]
    reasoning_configuration: list[MigrationAdvice]
    structured_output: list[MigrationAdvice]
    token_efficiency_opportunities: list[MigrationAdvice]
    hallucination_control: list[MigrationAdvice]
    migration_risks: list[MigrationAdvice]
    validation_recommendations: list[MigrationAdvice]


class ValidationLevel(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class ValidationIssue(StrictModel):
    code: str
    level: ValidationLevel
    message: str
    source_path: str | None = None


class PromptValidationResult(StrictModel):
    schema_version: Literal["1"] = "1"
    valid: bool
    source_prompt_sha256: str
    target_model: str
    target_platform: str | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)


class CompatibilityState(StrEnum):
    DIRECTLY_COMPATIBLE = "directly_compatible"
    MECHANICALLY_CONVERTIBLE = "mechanically_convertible"
    SEMANTICALLY_DIFFERENT = "semantically_different"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class CompatibilityAssessment(StrictModel):
    concern: str
    state: CompatibilityState
    rationale: str
    locations: list[SourceLocation] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ParserAssumption(StrictModel):
    parser: str
    expectation: str
    strictness: Literal["syntax", "schema", "unknown"]
    locations: list[SourceLocation] = Field(default_factory=list)


class InvocationAnalysis(StrictModel):
    schema_version: Literal["1"] = "1"
    application_root: str
    providers: list[str]
    platforms: list[str]
    sdk_imports: list[str]
    invocation_apis: list[str]
    model_identifiers: list[str]
    prompt_fields: list[str]
    parameters: dict[str, str | bool | int | float | None]
    streaming: bool
    reasoning_controls: list[str]
    multimodal_inputs: list[str]
    multimodal_contracts: list[MultimodalInputContract]
    tool_fields: list[str]
    structured_output_fields: list[str]
    response_parsers: list[str]
    parser_assumptions: list[ParserAssumption]
    tool_definitions: list[ToolDefinition]
    structured_outputs: list[StructuredOutputContract]
    tool_compatibility: CompatibilityAssessment | None = None
    structured_output_compatibility: CompatibilityAssessment | None = None
    source_locations: list[SourceLocation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ParameterMapping(StrictModel):
    source_name: str
    target_name: str | None = None
    target_container: str | None = None
    value: str | bool | int | float | None = None
    state: CompatibilityState
    rationale: str


class TargetInvocation(StrictModel):
    provider: str
    platform: str
    sdk: str
    client: str
    operation: str
    model_field: str
    model_id: str
    message_field: str
    system_field: str | None = None
    parameter_container: str | None = None
    tool_field: str | None = None
    structured_output_field: str | None = None
    image_input_format: str | None = None
    document_input_format: str | None = None


class ToolSchemaMigration(StrictModel):
    """Reviewable provider-specific tool payload derived from a normalized definition."""

    name: str | None = None
    source_format: str
    target_platform: str
    target_field: str | None = None
    state: CompatibilityState
    source_location: SourceLocation
    target_definition: dict[str, Any] | None = None
    target_choice_field: str | None = None
    target_choice: str | dict[str, Any] | None = None
    review_notes: list[str] = Field(default_factory=list)


class StructuredOutputMigration(StrictModel):
    """Reviewable provider-specific output configuration derived from a normalized schema."""

    name: str | None = None
    source_format: str
    target_platform: str
    target_field: str | None = None
    state: CompatibilityState
    source_location: SourceLocation
    target_configuration: dict[str, Any] | None = None
    review_notes: list[str] = Field(default_factory=list)


class PlatformConfigurationMigration(StrictModel):
    """Credential-safe target client/configuration contract for a platform migration."""

    source_providers: list[str]
    source_platforms: list[str]
    target_provider: str
    target_platform: str
    target_sdk: str
    target_client: str
    credential_strategy: Literal["environment_variable", "aws_default_chain", "unknown"]
    credential_environment_variables: list[str] = Field(default_factory=list)
    region_environment_variables: list[str] = Field(default_factory=list)
    state: CompatibilityState
    source_locations: list[SourceLocation] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


class InvocationMigrationSpec(StrictModel):
    schema_version: Literal["1", "2"] = "2"
    source_model: str
    target_model: str
    source_analysis: InvocationAnalysis
    target: TargetInvocation
    parameter_mappings: list[ParameterMapping]
    tool_schema_migrations: list[ToolSchemaMigration] = Field(default_factory=list)
    structured_output_migrations: list[StructuredOutputMigration] = Field(default_factory=list)
    configuration_migration: PlatformConfigurationMigration | None = None
    required_changes: list[str]
    warnings: list[str]
    blockers: list[str]


class MigrationPlan(StrictModel):
    """Canonical application-level V0.4 migration manifest."""

    schema_version: Literal["2"] = "2"
    source: MigrationEndpoint
    target: MigrationEndpoint
    application: MigrationApplicationSummary
    target_selection_rationale: list[str]
    model_differences: ModelComparison
    affected_files: list[str]
    required_changes: list[PlannedMigrationChange]
    optional_changes: list[PlannedMigrationChange]
    blockers: list[str]
    warnings: list[str]
    unknowns: list[str]
    prompt_changes: list[PromptMigrationSpec]
    invocation_changes: list[InvocationMigrationSpec]
    tool_changes: list[PlannedMigrationChange]
    output_contract_changes: list[PlannedMigrationChange]
    configuration_changes: list[PlannedMigrationChange]
    validation_results: list[ValidationIssue]
    required_tests: list[str]
    rollout_recommendations: list[str]
    migration_complexity: Literal["low", "medium", "high", "blocked"]
    overall_migration_risk: ComparisonSeverity


class MigrationEndpoint(StrictModel):
    model: str
    provider: str
    platform: str
    model_id: str


class MigrationApplicationSummary(StrictModel):
    root: str
    files_scanned: int
    finding_count: int
    providers: list[str]
    platforms: list[str]
    invocation_count: int
    prompt_count: int


class PlannedMigrationChange(StrictModel):
    category: Literal[
        "prompt",
        "invocation",
        "parameter",
        "tool",
        "output_contract",
        "configuration",
        "multimodal",
        "error_handling",
    ]
    description: str
    files: list[str] = Field(default_factory=list)
    locations: list[SourceLocation] = Field(default_factory=list)
