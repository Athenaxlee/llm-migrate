"""Validated domain models used across the registry, CLI, and MCP server."""

from __future__ import annotations

import hashlib
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


class InvocationSelector(StrictModel):
    """One named, reviewed way to invoke this platform representation on demand.

    `model_id` is the full id the platform accepts for invocation (e.g. the
    regional inference-profile id `us.anthropic.claude-sonnet-5`); `name` is
    its short selector label (e.g. `us`).
    """

    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    description: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)


class PlatformInvocation(StrictModel):
    """Reviewed facts about how this representation is invoked on demand.

    Fail-open: a platform representation without this block behaves exactly
    as before (the bare `model_id` is treated as the invocation id, with a
    report warning). Only a block that states `bare_on_demand_supported:
    false` hard-requires an invocation selector.
    """

    bare_on_demand_supported: bool | None = None
    selectors: list[InvocationSelector] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_invocation(self) -> PlatformInvocation:
        names = [selector.name for selector in self.selectors]
        if len(names) != len(set(names)):
            raise ValueError("invocation selector names must be unique")
        ids = [selector.model_id for selector in self.selectors]
        if len(ids) != len(set(ids)):
            raise ValueError("invocation selector model ids must be unique")
        if self.bare_on_demand_supported is False and not self.selectors:
            raise ValueError(
                "a platform that does not support bare on-demand invocation must "
                "declare at least one invocation selector"
            )
        return self


class PlatformAvailability(StrictModel):
    platform: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    endpoint: str | None = None
    regions: list[str] = Field(default_factory=list)
    capability_overrides: CapabilityOverrides | None = None
    invocation: PlatformInvocation | None = None
    sources: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selectors_differ_from_bare_id(self) -> PlatformAvailability:
        if self.invocation is not None and any(
            selector.model_id == self.model_id for selector in self.invocation.selectors
        ):
            raise ValueError(
                "an invocation selector model id must differ from the platform's bare model id"
            )
        return self


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


# The boolean capability fields of ModelCapabilities/CapabilityOverrides; the
# single home for every consumer that gates on "is this a boolean capability".
BOOLEAN_CAPABILITY_FIELDS = frozenset(
    {
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
    }
)


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
    INVOCATION_SELECTOR = "invocation_selector"


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


class ModelMatchStatus(StrEnum):
    RESOLVED = "resolved"
    NEEDS_CONFIRMATION = "needs_confirmation"
    NOT_FOUND = "not_found"


class ModelMatchCandidate(StrictModel):
    """One registry profile that plausibly matches a user-supplied identifier."""

    canonical_name: str
    display_name: str
    provider: str
    matched_identifier: str
    platform: str | None = None
    endpoint: str | None = None
    similarity: float = Field(ge=0, le=1)
    reason: str


class ModelMatchResult(StrictModel):
    """Registry-first identifier matching that never guesses silently.

    `resolved` carries an exact or deterministically equivalent match;
    `needs_confirmation` carries ranked candidates the user must confirm;
    `not_found` means the identifier is genuinely absent from the registry.
    """

    schema_version: Literal["1"] = "1"
    query: str
    platform_query: str | None = None
    status: ModelMatchStatus
    canonical_name: str | None = None
    platform: str | None = None
    model_id: str | None = None
    resolution: ResolvedModel | None = None
    candidates: list[ModelMatchCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    guidance: str


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


class SourceLocation(StrictModel):
    path: str
    line: int = Field(ge=1)
    column: int = Field(ge=0)
    end_line: int | None = Field(default=None, ge=1)


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
    locations: list[SourceLocation] = Field(default_factory=list)
    source_evidence_url: str | None = None
    target_evidence_url: str | None = None
    # Usage a prompt-guidance knowledge item is conditional on (additive,
    # v1.6.0-c); see MigrationKnowledgeItem.applies_when.
    applies_when: Literal["sampling", "reasoning", "structured_output", "tool_use"] | None = None


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


class PromptSourceFormat(StrEnum):
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    TEXT = "text"
    MARKDOWN = "markdown"


class PromptSourceConfidence(StrEnum):
    """How strongly the evidence ties a file to actual LLM prompt consumption."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PromptSourceOrigin(StrEnum):
    DISCOVERED = "discovered"
    OVERRIDE = "override"


class PromptComponent(StrictModel):
    """One addressable prompt inside a prompt source document."""

    role: Literal["system", "user", "developer", "unknown"] = "unknown"
    key: str | None = None
    content: str


class PromptSource(StrictModel):
    """A prompt-bearing file plus the evidence chain that classified it."""

    path: str
    format: PromptSourceFormat
    components: list[PromptComponent] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    confidence: PromptSourceConfidence
    origin: PromptSourceOrigin = PromptSourceOrigin.DISCOVERED
    # Model ids declared by the config profiles that reference this file;
    # empty when any referencing profile declares none (the file is then not
    # scopable to a model). Planning excludes a file referenced only by
    # profiles for other models.
    profile_model_ids: list[str] = Field(default_factory=list)


class PromptDiscoveryCoverage(StrEnum):
    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class PromptDiscoverySummary(StrictModel):
    """How completely detected prompt consumers map to static prompt content."""

    consumers: int = Field(default=0, ge=0)
    inline_consumers: int = Field(default=0, ge=0)
    source_backed_consumers: int = Field(default=0, ge=0)
    dynamic_consumers: int = Field(default=0, ge=0)
    # Consumers the user dismissed as genuinely runtime-built content
    # (additive, v1.6.0-a); they no longer count as dynamic.
    dismissed_consumers: int = Field(default=0, ge=0)
    resolved_sources: int = Field(default=0, ge=0)
    low_confidence_sources: int = Field(default=0, ge=0)
    coverage: PromptDiscoveryCoverage = PromptDiscoveryCoverage.RESOLVED


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
    schema_version: Literal["3"] = "3"
    root: str
    files_scanned: int
    findings: list[ApplicationFinding]
    requirements: ApplicationRequirements
    tool_definitions: list[ToolDefinition] = Field(default_factory=list)
    structured_outputs: list[StructuredOutputContract] = Field(default_factory=list)
    multimodal_inputs: list[MultimodalInputContract] = Field(default_factory=list)
    prompt_sources: list[PromptSource] = Field(default_factory=list)
    prompt_discovery: PromptDiscoverySummary = Field(default_factory=PromptDiscoverySummary)
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
    evidence_urls: list[str] = Field(default_factory=list)
    # The application usage this advice is conditional on (additive, v1.5.2):
    # registry guidance about structured output, tool use, or reasoning only
    # applies where the prompt or application exhibits it.
    applies_when: Literal["structured_output", "tool_use", "reasoning"] | None = None


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
    schema_version: Literal["4"] = "4"
    source_model: str
    target_model: str
    source_provider: str
    target_provider: str
    source_platform: str | None = None
    target_platform: str | None = None
    source_prompt_sha256: str
    source_path: str | None = None
    source_component: str | None = None
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


class BlockerCategory(StrEnum):
    CAPABILITY = "capability"
    SOURCE_CONSISTENCY = "source_consistency"
    PLATFORM_AMBIGUITY = "platform_ambiguity"
    CONTEXT_WINDOW = "context_window"
    INVALID_SCHEMA = "invalid_schema"
    PARAMETER = "parameter"
    INVOCATION = "invocation"
    OTHER = "other"


class MigrationBlocker(StrictModel):
    """One structured migration blocker with a stable identity and its evidence.

    The `id` is deterministic over (category, code, message, discriminator), so
    the same blocker keeps the same id across plan regenerations and a recorded
    decision can keep matching it; when the underlying facts change, the id
    changes and dependent decisions surface as stale instead of misapplying.
    """

    id: str
    code: str
    category: BlockerCategory
    message: str
    evidence_urls: list[str] = Field(default_factory=list)
    locations: list[SourceLocation] = Field(default_factory=list)
    data: dict[str, Any] | None = None

    @property
    def rendered(self) -> str:
        """Stable human-readable view used by reports and worklists."""
        text = f"{self.message} [{self.category.value}; id: {self.id}]"
        paths = sorted({item.path for item in self.locations})
        if paths:
            shown = ", ".join(paths[:3]) + (f" +{len(paths) - 3} more" if len(paths) > 3 else "")
            text += f" (files: {shown})"
        if self.evidence_urls:
            text += f" (evidence: {self.evidence_urls[0]})"
        return text


def migration_blocker(
    code: str,
    category: BlockerCategory,
    message: str,
    *,
    evidence_urls: list[str] | None = None,
    locations: list[SourceLocation] | None = None,
    data: dict[str, Any] | None = None,
    discriminator: str = "",
) -> MigrationBlocker:
    """Build a blocker with its deterministic stable id."""
    digest = hashlib.sha256(
        f"{category.value}|{code}|{message}|{discriminator}".encode()
    ).hexdigest()[:10]
    return MigrationBlocker(
        id=f"{code}:{digest}",
        code=code,
        category=category,
        message=message,
        evidence_urls=evidence_urls or [],
        locations=locations or [],
        data=data,
    )


RUN_DIR_PLACEHOLDER = "<run_dir>"


class UnknownKind(StrEnum):
    PROMPT_CANDIDATE = "prompt_candidate"
    DYNAMIC_PROMPT_CONSUMER = "dynamic_prompt_consumer"
    COMPATIBILITY = "compatibility"
    MODEL_DIFFERENCE = "model_difference"
    CONTESTED_EVIDENCE = "contested_evidence"


class MigrationUnknown(StrictModel):
    """One unresolved (or closed) migration unknown that gives a direction.

    Every unknown says what it is about (`subject`), why it matters for THIS
    application, the concrete next `action` (a ready-to-run tool call or CLI
    command with its arguments filled; in a guided run the run directory is
    substituted for `RUN_DIR_PLACEHOLDER`), and the condition that closes it.
    The `id` is deterministic over (kind, discriminator), so a recorded
    observation keeps matching across plan regenerations. `message` keeps the
    pre-v5 one-line text as the stable rendered view. Unknowns the scan
    already answers are kept `closed_by_scan`, and ones a recorded user
    action answered are `closed_by_action`, each with its reason, so the
    report can say how every unknown was closed.
    """

    id: str
    kind: UnknownKind
    subject: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    action: str = Field(min_length=1)
    closing_condition: str = Field(min_length=1)
    message: str = Field(min_length=1)
    status: Literal["open", "closed_by_scan", "closed_by_action"] = "open"
    closed_reason: str | None = None
    evidence_urls: list[str] = Field(default_factory=list)
    data: dict[str, Any] | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def rendered(self) -> str:
        """Stable one-line view used by worklists and status."""
        return f"{self.message} [{self.kind.value}; id: {self.id}] Action: {self.action}"


def unknown_id(kind: UnknownKind, discriminator: str) -> str:
    """Deterministic unknown id over its kind and identifying discriminator."""
    digest = hashlib.sha256(f"{kind.value}|{discriminator}".encode()).hexdigest()[:10]
    return f"{kind.value}:{digest}"


def open_unknowns(unknowns: list[MigrationUnknown]) -> list[MigrationUnknown]:
    return [item for item in unknowns if item.is_open]


def dedupe_blockers(blockers: list[MigrationBlocker]) -> list[MigrationBlocker]:
    """Stable-id deduplication with a deterministic report order.

    Duplicates merge their locations and evidence instead of being dropped, so
    a blocker produced at several code sites keeps every site — the injected
    redesign task and the report must never under-state the affected files.
    """
    unique: dict[str, MigrationBlocker] = {}
    for blocker in blockers:
        existing = unique.get(blocker.id)
        if existing is None:
            unique[blocker.id] = blocker
            continue
        locations = list(existing.locations)
        locations.extend(item for item in blocker.locations if item not in locations)
        evidence = list(existing.evidence_urls)
        evidence.extend(item for item in blocker.evidence_urls if item not in evidence)
        unique[blocker.id] = existing.model_copy(
            update={
                "locations": sorted(locations, key=lambda item: (item.path, item.line)),
                "evidence_urls": evidence,
            }
        )
    return sorted(unique.values(), key=lambda item: (item.category.value, item.code, item.message))


PROMPT_ISSUE_CATEGORIES = {
    "context_window_exceeded": BlockerCategory.CONTEXT_WINDOW,
    "empty_prompt": BlockerCategory.OTHER,
    "invalid_target_platform": BlockerCategory.PLATFORM_AMBIGUITY,
}


def prompt_issue_blocker(
    issue: ValidationIssue,
    *,
    evidence_urls: list[str] | None = None,
    locations: list[SourceLocation] | None = None,
    data: dict[str, Any] | None = None,
) -> MigrationBlocker:
    """The one derivation of a blocker (and its id) from a prompt validation issue.

    Both plan generation and the submission gate must derive the SAME id from
    the same issue, or recorded accept decisions could not be matched back to
    the validation blocker they accepted.
    """
    return migration_blocker(
        code=issue.code,
        category=PROMPT_ISSUE_CATEGORIES.get(issue.code, BlockerCategory.OTHER),
        message=issue.message,
        evidence_urls=evidence_urls,
        locations=locations,
        data=data,
        discriminator=issue.source_path or "",
    )


def migration_complexity(
    blockers: list[MigrationBlocker],
    highest_severity: ComparisonSeverity,
    required_changes: list[Any],
    warnings: list[str],
) -> Literal["low", "medium", "high", "blocked"]:
    """The single complexity ladder shared by plan generation and decision replay."""
    if blockers:
        return "blocked"
    if highest_severity in {ComparisonSeverity.HIGH, ComparisonSeverity.BREAKING}:
        return "high"
    if required_changes or warnings:
        return "medium"
    return "low"


class ResolutionKind(StrEnum):
    RETARGET = "retarget"
    REDESIGN_TASK = "redesign_task"
    CORRECTION = "correction"
    ACCEPT_WITH_RATIONALE = "accept_with_rationale"


class EndpointChange(StrictModel):
    """A partial source/target identity change; unset fields keep their value."""

    model: str | None = None
    platform: str | None = None
    endpoint: str | None = None
    invocation_selector: str | None = None


class RedesignTaskSpec(StrictModel):
    """The required adaptation task a redesign decision injects into the plan."""

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
    guidance: list[str] = Field(default_factory=list)
    evidence_urls: list[str] = Field(default_factory=list)


class BlockerDecision(StrictModel):
    """One durable, auditable user decision about one migration blocker."""

    schema_version: Literal["1"] = "1"
    blocker_id: str
    blocker_code: str
    blocker_message: str
    option_id: str
    kind: ResolutionKind
    summary: str
    rationale: str
    decided_on: date
    target_change: EndpointChange | None = None
    source_change: EndpointChange | None = None
    task: RedesignTaskSpec | None = None


class AppliedBlockerDecision(StrictModel):
    """How one recorded decision affected the current plan regeneration.

    `superseded` marks a retarget/correction that was honored and later
    replaced by a newer decision of the same kind: recorded history, not a
    stale alarm.
    """

    decision: BlockerDecision
    status: Literal["applied", "stale", "superseded"]
    detail: str


class InvocationMigrationSpec(StrictModel):
    schema_version: Literal["3"] = "3"
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
    blockers: list[MigrationBlocker]


class MigrationPlan(StrictModel):
    """Canonical application-level V0.4 migration manifest.

    Schema version 5 (v1.6.0-b) types `unknowns` as `MigrationUnknown`
    records (subject, why it matters, action, closing condition, status);
    each record's `message` keeps the pre-v5 rendered text.
    """

    schema_version: Literal["5"] = "5"
    source: MigrationEndpoint
    target: MigrationEndpoint
    application: MigrationApplicationSummary
    prompt_discovery: PromptDiscoverySummary = Field(default_factory=PromptDiscoverySummary)
    target_selection_rationale: list[str]
    model_differences: ModelComparison
    affected_files: list[str]
    incidental_files: list[str] = Field(default_factory=list)
    required_changes: list[PlannedMigrationChange]
    optional_changes: list[PlannedMigrationChange]
    blockers: list[MigrationBlocker]
    decisions: list[AppliedBlockerDecision] = Field(default_factory=list)
    warnings: list[str]
    unknowns: list[MigrationUnknown]
    # Prompt files the scan shows this migration does not govern (additive,
    # v1.5.2): never prepared, never tasks, never unknowns.
    out_of_scope: list[str] = Field(default_factory=list)
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
