"""Deterministic blocker resolution: questions, evidence-backed options, decisions.

The toolkit never chooses. For every unresolved blocker it derives, from
registry facts alone, the question the host agent must ask the user, plus
concrete options (retarget / redesign / correction / accept-with-rationale)
with their consequences, backing evidence, and the exact machine-actionable
next step. User choices are recorded durably in the run's ``decisions.yaml``
and re-applied on every plan regeneration; a decision whose blocker no longer
exists is reported as stale, never silently applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

from llm_migrate.analyzers.invocation import (
    ADAPTER_PLATFORMS,
    STRUCTURED_OUTPUT_FIELD_PLATFORMS,
)
from llm_migrate.core.agent_research import ModelEndpointIdentity
from llm_migrate.core.comparison import capability_evidence_url, evidence_url
from llm_migrate.core.models import (
    BOOLEAN_CAPABILITY_FIELDS,
    ApplicationAnalysis,
    AppliedBlockerDecision,
    BlockerCategory,
    BlockerDecision,
    EndpointChange,
    MigrationBlocker,
    MigrationPlan,
    ModelProfile,
    PlannedMigrationChange,
    PlatformAvailability,
    PromptValidationResult,
    RecommendationConstraints,
    RedesignTaskSpec,
    ResolutionKind,
    StrictModel,
    ValidationLevel,
    migration_complexity,
    prompt_issue_blocker,
)
from llm_migrate.core.recommendation import recommend_models
from llm_migrate.core.registry import ModelRegistry, RegistryError
from llm_migrate.core.resolver import effective_capabilities, resolve_model
from llm_migrate.core.runstate import atomic_write_text
from llm_migrate.core.workspace import MigrationRunConfig, WorkspaceError

DECISIONS_FILENAME = "decisions.yaml"

RESOLUTION_GUIDANCE = [
    "Present each blocker's question, options, consequences, and evidence VERBATIM "
    "to the user, one blocker at a time; the user is the decision maker.",
    "Never choose an option on the user's behalf, never invent options, and never "
    "retry submissions to make a blocker disappear; only recorded decisions resolve "
    "blockers.",
    "Record each user answer with record_blocker_decision "
    "(CLI: `llm-migrate run decide`). The accept option requires the user's own "
    "free-text rationale and is never a default.",
    "Decisions persist in decisions.yaml and are re-applied on every plan "
    "regeneration; a decision whose blocker no longer exists is reported as stale, "
    "not silently applied.",
    "If a blocker lists no registry-backed option, tell the user exactly that: "
    "fixing the application or an explicit accept-with-rationale are the only paths.",
]


class ResolutionNextStep(StrictModel):
    """The exact tool call to make once the user chooses this option."""

    tool: str
    arguments: dict[str, Any]


class ResolutionOption(StrictModel):
    """One evidence-backed way to resolve a blocker; never chosen automatically."""

    id: str
    kind: ResolutionKind
    summary: str
    consequences: list[str] = Field(default_factory=list)
    evidence_urls: list[str] = Field(default_factory=list)
    next_step: ResolutionNextStep
    target_change: EndpointChange | None = None
    source_change: EndpointChange | None = None
    task: RedesignTaskSpec | None = None


class BlockerResolution(StrictModel):
    """The question to ask the user for one blocker, with its options."""

    blocker: MigrationBlocker
    question: str
    options: list[ResolutionOption]
    no_registry_backed_option: str | None = None


class BlockerResolutionSet(StrictModel):
    """Everything the host agent needs to drive blockers to user decisions."""

    schema_version: Literal["1"] = "1"
    run_id: str
    resolutions: list[BlockerResolution] = Field(default_factory=list)
    decisions: list[AppliedBlockerDecision] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


class DecisionLog(StrictModel):
    """Durable record of every blocker decision in one run (decisions.yaml)."""

    schema_version: Literal["1"] = "1"
    run_id: str
    decisions: list[BlockerDecision] = Field(default_factory=list)


class BlockerDecisionResult(StrictModel):
    """Outcome of recording one decision, with the refreshed blocker state."""

    schema_version: Literal["1"] = "1"
    accepted: bool
    problems: list[str] = Field(default_factory=list)
    decision: BlockerDecision | None = None
    run_config_updated: bool = False
    unresolved_blockers: list[str] = Field(default_factory=list)
    stale_decisions: list[str] = Field(default_factory=list)
    message: str
    next_steps: list[str] = Field(default_factory=list)


def load_decision_log(run_dir: Path, run_id: str) -> DecisionLog:
    path = Path(run_dir) / DECISIONS_FILENAME
    if not path.is_file():
        return DecisionLog(run_id=run_id)
    try:
        log = DecisionLog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise WorkspaceError(f"invalid decision log {path}: {exc}") from exc
    if log.run_id != run_id:
        # Blocker ids are content-derived and identical across runs of the
        # same application, so a copied decisions.yaml would silently apply
        # another run's decisions. Refuse instead.
        raise WorkspaceError(
            f"decision log {path} belongs to run {log.run_id!r}, not {run_id!r}; "
            "decisions never transfer between runs — delete the copied "
            "decisions.yaml or decide this run's blockers explicitly"
        )
    return log


def save_decision_log(run_dir: Path, log: DecisionLog) -> Path:
    path = Path(run_dir) / DECISIONS_FILENAME
    atomic_write_text(path, yaml.safe_dump(log.model_dump(mode="json"), sort_keys=False))
    return path


def upsert_decision(log: DecisionLog, decision: BlockerDecision) -> DecisionLog:
    kept = [item for item in log.decisions if item.blocker_id != decision.blocker_id]
    return log.model_copy(update={"decisions": [*kept, decision]})


# ---------------------------------------------------------------------------
# Option derivation (registry facts only; nothing is invented)
# ---------------------------------------------------------------------------


def _profile_evidence(profile: ModelProfile) -> list[str]:
    url = next((str(source.url) for source in profile.sources if source.url), None)
    return [url] if url else []


def _platform_evidence(profile: ModelProfile, platform: PlatformAvailability) -> list[str]:
    url = evidence_url(profile, platform.sources, "platforms")
    return [url] if url else []


def _capability_evidence(
    profile: ModelProfile, platform: PlatformAvailability | None, field: str
) -> list[str]:
    url = capability_evidence_url(profile, platform, field)
    return [url] if url else []


def _next_step(
    run_dir: str, blocker: MigrationBlocker, option_id: str, kind: ResolutionKind
) -> ResolutionNextStep:
    arguments: dict[str, Any] = {
        "run_dir": run_dir,
        "blocker_id": blocker.id,
        "option_id": option_id,
    }
    # The accept option deliberately carries no rationale value: the required
    # rationale is the user's own words, never a template an agent could echo.
    return ResolutionNextStep(tool="record_blocker_decision", arguments=arguments)


def _accept_option(run_dir: str, blocker: MigrationBlocker) -> ResolutionOption:
    return ResolutionOption(
        id="accept",
        kind=ResolutionKind.ACCEPT_WITH_RATIONALE,
        summary="Accept this risk and continue.",
        consequences=[
            "The report records it as an accepted risk; it no longer blocks the run.",
            "You must give your own reason. This option is never a default and the agent "
            "may not choose it for you.",
        ],
        next_step=_next_step(run_dir, blocker, "accept", ResolutionKind.ACCEPT_WITH_RATIONALE),
    )


def _endpoint_suffix(endpoint: str | None) -> str:
    return f" (endpoint {endpoint})" if endpoint else ""


def _retarget_same_model(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    capability: str | None,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    """Other registry representations of the same target where the fact holds."""
    try:
        profile = registry.get(config.target.model)
    except RegistryError:
        return []
    label = capability.replace("_", " ") if capability else None
    options: list[ResolutionOption] = []
    for representation in profile.platforms:
        if (
            representation.platform == config.target.platform
            and representation.endpoint == config.target.endpoint
        ):
            continue
        if representation.platform not in ADAPTER_PLATFORMS:
            continue
        if capability is not None:
            capabilities = effective_capabilities(profile, representation)
            if getattr(capabilities, capability, None) is not True:
                continue
            if (
                capability == "structured_output"
                and representation.platform not in STRUCTURED_OUTPUT_FIELD_PLATFORMS
            ):
                continue
            evidence = _capability_evidence(profile, representation, capability)
            supported = f"the registry records native {label} as supported"
        else:
            evidence = _platform_evidence(profile, representation)
            supported = "a deterministic invocation adapter exists for it"
        option_id = (
            "retarget-"
            + representation.platform
            + (f"-{representation.endpoint}" if representation.endpoint else "")
        )
        consequences = [
            f"The target becomes {config.target.model} on {representation.platform}"
            f"{_endpoint_suffix(representation.endpoint)} (model id "
            f"{representation.model_id}); the plan is rebuilt for it.",
        ]
        if representation.platform != config.target.platform:
            consequences.append(
                f"Credentials, configuration, and request shape move from "
                f"{config.target.platform} to {representation.platform}."
            )
        options.append(
            ResolutionOption(
                id=option_id,
                kind=ResolutionKind.RETARGET,
                summary=(
                    f"Keep {config.target.model} but run it on {representation.platform}"
                    f"{_endpoint_suffix(representation.endpoint)}, where {supported}."
                ),
                consequences=consequences,
                evidence_urls=evidence,
                next_step=_next_step(run_dir, blocker, option_id, ResolutionKind.RETARGET),
                target_change=EndpointChange(
                    model=config.target.model,
                    platform=representation.platform,
                    endpoint=representation.endpoint,
                ),
            )
        )
    return options


def _alternative_model_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    analysis: ApplicationAnalysis,
    blocker: MigrationBlocker,
    run_dir: str,
    *,
    required_capability: str | None = None,
    minimum_context_window: int | None = None,
) -> list[ResolutionOption]:
    """Different target models via the existing recommendation engine."""
    constraints = RecommendationConstraints(
        platform=config.target.platform,
        required_capabilities=(
            {required_capability} if required_capability in BOOLEAN_CAPABILITY_FIELDS else set()
        ),
        minimum_context_window=minimum_context_window,
        source_model=config.source.model,
        source_platform=config.source.platform,
    )
    try:
        result = recommend_models(registry, constraints, requirements=analysis.requirements)
    except RegistryError:
        return []
    options: list[ResolutionOption] = []
    for recommendation in result.recommendations:
        if len(options) == 2:
            break
        if recommendation.canonical_name == config.target.model:
            continue
        if recommendation.evidence_state == "fixture":
            continue
        profile = registry.get(recommendation.canonical_name)
        representation = next(
            (item for item in profile.platforms if item.platform == recommendation.platform),
            None,
        )
        if required_capability:
            evidence = _capability_evidence(profile, representation, required_capability)
            satisfies = f"native {required_capability.replace('_', ' ')}"
        elif minimum_context_window:
            evidence = _capability_evidence(profile, representation, "context_window_tokens")
            satisfies = f"a context window of at least {minimum_context_window} tokens"
        else:
            evidence = _profile_evidence(profile)
            satisfies = "the detected application requirements"
        option_id = f"alternative-{recommendation.canonical_name}"
        options.append(
            ResolutionOption(
                id=option_id,
                kind=ResolutionKind.RETARGET,
                summary=(
                    f"Switch to a different model: {recommendation.canonical_name} on "
                    f"{recommendation.platform}. The registry records it as providing "
                    f"{satisfies}, and it meets every requirement the scan found."
                ),
                consequences=[
                    "A different model: prompts, parameters, and behavior must be "
                    "re-planned and re-tested.",
                    *(f"Recommendation tradeoff: {item}" for item in recommendation.tradeoffs[:2]),
                ],
                evidence_urls=evidence,
                next_step=_next_step(run_dir, blocker, option_id, ResolutionKind.RETARGET),
                target_change=EndpointChange(
                    model=recommendation.canonical_name,
                    platform=recommendation.platform,
                    endpoint=representation.endpoint if representation else None,
                ),
            )
        )
    return options


def _redesign_option(
    run_dir: str,
    blocker: MigrationBlocker,
    task: RedesignTaskSpec,
    summary: str,
    consequences: list[str],
) -> ResolutionOption:
    return ResolutionOption(
        id="redesign",
        kind=ResolutionKind.REDESIGN_TASK,
        summary=summary,
        consequences=[
            *consequences,
            "This adds a required task (with evidence-linked guidance) to the plan for as "
            "long as the decision stands.",
        ],
        evidence_urls=task.evidence_urls,
        next_step=_next_step(run_dir, blocker, "redesign", ResolutionKind.REDESIGN_TASK),
        task=task,
    )


def _capability_redesign(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    plan: MigrationPlan,
    capability: str,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    """Registry-grounded redesign off the blocked native feature, when one exists."""
    try:
        profile = registry.get(config.target.model)
    except RegistryError:
        return []
    if capability == "structured_output":
        parsers = sorted(
            {
                assumption.parser
                for invocation in plan.invocation_changes
                for assumption in invocation.source_analysis.parser_assumptions
            }
        )
        guidance = [
            "Remove the native structured-output request configuration and state the "
            "output contract in explicit prompt instructions (the JSON Schema, required "
            "fields, and a JSON-only output requirement).",
            (
                "Keep validating responses with the application's existing parser(s): "
                + ", ".join(parsers)
                + "."
                if parsers
                else "Add explicit response validation in the application; no response "
                "parser was statically detected."
            ),
            *profile.prompt_guidance.structured_output_guidance,
        ]
        guidance_url = evidence_url(profile, profile.prompt_guidance.sources, "prompt_guidance")
        task = RedesignTaskSpec(
            category="output_contract",
            description=(
                "Redesign off native structured output: enforce the output contract "
                "through prompt instructions plus the existing application parser "
                f"instead of the unsupported native feature on {config.target.platform}."
            ),
            guidance=guidance,
            evidence_urls=[*blocker.evidence_urls, *([guidance_url] if guidance_url else [])],
        )
        return [
            _redesign_option(
                run_dir,
                blocker,
                task,
                summary=(
                    "Stop using native structured output: enforce JSON through the prompt "
                    "and keep the existing parser."
                ),
                consequences=[
                    "The platform no longer guarantees the output format; bad outputs show "
                    "up at your parser instead of the API.",
                ],
            )
        ]
    if capability == "streaming":
        task = RedesignTaskSpec(
            category="invocation",
            description=(
                "Redesign off streaming: replace streaming consumption with a "
                "non-streaming request/response flow for "
                f"{config.target.model} on {config.target.platform}."
            ),
            guidance=[
                "Remove stream=True and incremental chunk handling; read the complete "
                "response instead.",
            ],
            evidence_urls=blocker.evidence_urls,
        )
        return [
            _redesign_option(
                run_dir,
                blocker,
                task,
                summary="Stop streaming and read complete responses.",
                consequences=[
                    "The first output arrives later, and incremental display is lost.",
                ],
            )
        ]
    return []


def _capability_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    plan: MigrationPlan,
    analysis: ApplicationAnalysis,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    capability = (blocker.data or {}).get("capability")
    if blocker.code == "no_invocation_adapter":
        return _retarget_same_model(registry, config, None, blocker, run_dir)
    if not isinstance(capability, str):
        return []
    options = _retarget_same_model(registry, config, capability, blocker, run_dir)
    # An alternative model keeps the current platform, so it can only help
    # when the platform itself is not what blocks: never for a missing
    # deterministic adapter mapping, and never for structured output on a
    # platform whose adapter has no structured-output field.
    platform_can_carry = not blocker.code.endswith("_mapping_missing") and (
        capability != "structured_output"
        or config.target.platform in STRUCTURED_OUTPUT_FIELD_PLATFORMS
    )
    if platform_can_carry:
        options.extend(
            _alternative_model_options(
                registry,
                config,
                analysis,
                blocker,
                run_dir,
                required_capability=capability,
            )
        )
    options.extend(_capability_redesign(registry, config, plan, capability, blocker, run_dir))
    return options


def _consistency_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    data = blocker.data or {}
    detected = [item for item in data.get("detected", []) if isinstance(item, str)]
    detected_platforms = [
        item for item in data.get("detected_platforms", []) if isinstance(item, str)
    ]
    options: list[ResolutionOption] = []
    if blocker.code == "source_model_mismatch":
        for name in detected[:3]:
            try:
                profile = registry.get(name)
            except RegistryError:
                continue
            # Pair the corrected model with a concrete representation on the
            # platform the scan detected (never the mistakenly declared one);
            # the endpoint must come along or a multi-endpoint platform would
            # make the tool's own option unrecordable (ambiguous resolution).
            representation = next(
                (item for item in profile.platforms if item.platform in detected_platforms),
                None,
            )
            if representation is None and len(profile.platforms) == 1:
                representation = profile.platforms[0]
            option_id = f"correct-source-model-{name}"
            consequences = [
                "The run's source model is corrected to the one the scan detected, and the "
                "plan is rebuilt.",
                "Equivalent to restarting with start_migration(application_path="
                f"{config.application_root!r}, source={name!r}, "
                f"target={config.target.model!r}, "
                f"target_platform={config.target.platform!r}).",
            ]
            if len(detected) > 1:
                consequences.append(
                    f"The application uses several source models; this run then covers only "
                    f"{name}. Start a separate run for each other model."
                )
            options.append(
                ResolutionOption(
                    id=option_id,
                    kind=ResolutionKind.CORRECTION,
                    summary=f"Correct the source model to {name}, the one found in the code.",
                    consequences=consequences,
                    evidence_urls=_profile_evidence(profile),
                    next_step=_next_step(run_dir, blocker, option_id, ResolutionKind.CORRECTION),
                    source_change=EndpointChange(
                        model=name,
                        platform=representation.platform if representation else None,
                        endpoint=representation.endpoint if representation else None,
                    ),
                )
            )
    if blocker.code == "source_platform_mismatch":
        source_profile: ModelProfile | None
        try:
            source_profile = registry.get(config.source.model)
        except RegistryError:
            source_profile = None
        for platform in detected[:3]:
            representations = (
                [item for item in source_profile.platforms if item.platform == platform]
                if source_profile
                else []
            )
            for representation in representations[:2]:
                option_id = f"correct-source-platform-{platform}" + (
                    f"-{representation.endpoint}" if representation.endpoint else ""
                )
                options.append(
                    ResolutionOption(
                        id=option_id,
                        kind=ResolutionKind.CORRECTION,
                        summary=(
                            f"Correct the source platform to {platform}"
                            f"{_endpoint_suffix(representation.endpoint)}, the one found in "
                            "the code."
                        ),
                        consequences=[
                            "The run's source platform is corrected and the plan is rebuilt.",
                            "Equivalent to restarting with start_migration(application_path="
                            f"{config.application_root!r}, source={config.source.model!r}, "
                            f"source_platform={platform!r}, target={config.target.model!r}, "
                            f"target_platform={config.target.platform!r}).",
                        ],
                        evidence_urls=(
                            _platform_evidence(source_profile, representation)
                            if source_profile
                            else []
                        ),
                        next_step=_next_step(
                            run_dir, blocker, option_id, ResolutionKind.CORRECTION
                        ),
                        source_change=EndpointChange(
                            platform=platform, endpoint=representation.endpoint
                        ),
                    )
                )
    return options


def _platform_ambiguity_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    """Enumerate the concrete registry endpoints the target actually has."""
    try:
        profile = registry.get(config.target.model)
    except RegistryError:
        return []
    options: list[ResolutionOption] = []
    for representation in profile.platforms:
        if (
            representation.platform == config.target.platform
            and representation.endpoint == config.target.endpoint
        ):
            continue
        option_id = (
            "retarget-"
            + representation.platform
            + (f"-{representation.endpoint}" if representation.endpoint else "")
        )
        options.append(
            ResolutionOption(
                id=option_id,
                kind=ResolutionKind.RETARGET,
                summary=(
                    f"Use exactly {representation.model_id} on {representation.platform}"
                    f"{_endpoint_suffix(representation.endpoint)}."
                ),
                consequences=[
                    "The target is pinned to this exact platform entry and the plan is rebuilt.",
                ],
                evidence_urls=_platform_evidence(profile, representation),
                next_step=_next_step(run_dir, blocker, option_id, ResolutionKind.RETARGET),
                target_change=EndpointChange(
                    model=config.target.model,
                    platform=representation.platform,
                    endpoint=representation.endpoint,
                ),
            )
        )
    return options[:4]


def _context_window_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    analysis: ApplicationAnalysis,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    data = blocker.data or {}
    required = data.get("required_context_window") or data.get("approximate_tokens")
    target_context = data.get("target_context_window")
    options: list[ResolutionOption] = []
    if isinstance(required, int):
        profile: ModelProfile | None
        try:
            profile = registry.get(config.target.model)
        except RegistryError:
            profile = None
        if profile is not None:
            for representation in profile.platforms:
                if (
                    representation.platform == config.target.platform
                    and representation.endpoint == config.target.endpoint
                ):
                    continue
                if representation.platform not in ADAPTER_PLATFORMS:
                    continue
                capabilities = effective_capabilities(profile, representation)
                window = capabilities.context_window_tokens
                if window is None or window < required:
                    continue
                option_id = (
                    "retarget-"
                    + representation.platform
                    + (f"-{representation.endpoint}" if representation.endpoint else "")
                )
                options.append(
                    ResolutionOption(
                        id=option_id,
                        kind=ResolutionKind.RETARGET,
                        summary=(
                            f"Keep {config.target.model} but run it on "
                            f"{representation.platform}"
                            f"{_endpoint_suffix(representation.endpoint)}, where the "
                            f"registry records a {window}-token context window "
                            f"(≥ {required} required)."
                        ),
                        consequences=[
                            "The target moves to this platform entry and the plan is rebuilt.",
                        ],
                        evidence_urls=_capability_evidence(
                            profile, representation, "context_window_tokens"
                        ),
                        next_step=_next_step(run_dir, blocker, option_id, ResolutionKind.RETARGET),
                        target_change=EndpointChange(
                            model=config.target.model,
                            platform=representation.platform,
                            endpoint=representation.endpoint,
                        ),
                    )
                )
        options.extend(
            _alternative_model_options(
                registry,
                config,
                analysis,
                blocker,
                run_dir,
                minimum_context_window=required,
            )
        )
    reduction_target = (
        f"the target's {target_context}-token context window"
        if isinstance(target_context, int)
        else "the target's context window"
    )
    needed = f" (the application currently needs ~{required} tokens)" if required else ""
    source_path = data.get("source_path")
    scope = f" in {source_path}" if isinstance(source_path, str) and source_path else ""
    task = RedesignTaskSpec(
        category="prompt",
        description=(f"Reduce the prompt content{scope} to fit {reduction_target}{needed}."),
        guidance=[
            "Deduplicate repeated instructions and trim examples; move static reference "
            "material out of the prompt and into retrieval or tool results.",
            "Preserve the prompt's detected semantic requirements; verify the reduction "
            "with a source-versus-target evaluation before rollout.",
        ],
        evidence_urls=blocker.evidence_urls,
    )
    options.append(
        _redesign_option(
            run_dir,
            blocker,
            task,
            summary="Reduce the prompt so it fits the target context window.",
            consequences=[
                "Prompt content is removed or restructured; re-test behavior on "
                "representative inputs.",
            ],
        )
    )
    return options


def _invalid_schema_options(blocker: MigrationBlocker, run_dir: str) -> list[ResolutionOption]:
    data = blocker.data or {}
    category: Literal["tool", "output_contract"] = (
        "tool" if data.get("capability") == "tool_use" else "output_contract"
    )
    location = blocker.locations[0] if blocker.locations else None
    where = f" at {location.path}:{location.line}" if location else ""
    task = RedesignTaskSpec(
        category=category,
        description=(
            f"Fix the invalid JSON Schema declared{where} so it validates against "
            "JSON Schema Draft 2020-12."
        ),
        guidance=[blocker.message],
    )
    return [
        _redesign_option(
            run_dir,
            blocker,
            task,
            summary=f"Fix the invalid JSON Schema{where} in the application.",
            consequences=[
                "The schema is the application's own contract; it must be valid before "
                "any target can enforce or convert it.",
            ],
        )
    ]


def _parameter_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    data = blocker.data or {}
    name = data.get("parameter")
    if not isinstance(name, str):
        return []
    registry_name = data.get("registry_name") or name
    notes = data.get("notes")
    evidence: list[str] = []
    try:
        profile = registry.get(config.target.model)
    except RegistryError:
        profile = None
    if profile is not None:
        support = profile.parameters.get(str(registry_name))
        if support is not None:
            url = evidence_url(profile, support.sources, "parameters")
            evidence = [url] if url else []
    task = RedesignTaskSpec(
        category="parameter",
        description=(
            f"Remove or replace the {name} parameter, which "
            f"{config.target.model} marks unsupported."
        ),
        guidance=[str(notes)] if notes else [],
        evidence_urls=evidence or blocker.evidence_urls,
    )
    return [
        _redesign_option(
            run_dir,
            blocker,
            task,
            summary=(f"Remove or replace the unsupported {name} parameter in the invocation."),
            consequences=[
                f"Requests stop sending {name}; re-test any behavior that depended on it.",
            ],
        )
    ]


_KIND_QUESTION_LABELS = {
    ResolutionKind.RETARGET: "run on a model or platform where this works (retarget)",
    ResolutionKind.REDESIGN_TASK: "change the application so it no longer needs this (redesign)",
    ResolutionKind.CORRECTION: "correct what the run declared (correction)",
    ResolutionKind.ACCEPT_WITH_RATIONALE: "accept the risk and give your reason (accept)",
}


def _question(blocker: MigrationBlocker, options: list[ResolutionOption]) -> str:
    kinds: list[str] = []
    for option in options:
        label = _KIND_QUESTION_LABELS[option.kind]
        if label not in kinds:
            kinds.append(label)
    return f"{blocker.message.rstrip('.')}. What do you want to do: " + "; or ".join(kinds) + "?"


def _invocation_options(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    blocker: MigrationBlocker,
    run_dir: str,
) -> list[ResolutionOption]:
    """One correction option per reviewed invocation selector of the target."""
    try:
        resolved = resolve_model(
            registry, config.target.model, config.target.platform, config.target.endpoint
        )
    except RegistryError:
        return []
    platform = resolved.platform
    if platform is None or platform.invocation is None:
        return []
    options: list[ResolutionOption] = []
    for selector in platform.invocation.selectors:
        evidence = [str(source.url) for source in selector.sources if source.url] or [
            str(source.url) for source in platform.invocation.sources if source.url
        ]
        option_id = f"selector-{selector.name}"
        options.append(
            ResolutionOption(
                id=option_id,
                kind=ResolutionKind.CORRECTION,
                summary=(
                    f"Call the target as {selector.model_id} (selector {selector.name!r})."
                    + (f" {selector.description}" if selector.description else "")
                ),
                consequences=[
                    f"Every adapted file must use {selector.model_id!r}; the worklist, "
                    "submission checks, and report enforce it.",
                    f"Files that use only the bare id {platform.model_id!r} are rejected.",
                ],
                evidence_urls=evidence[:2],
                next_step=_next_step(run_dir, blocker, option_id, ResolutionKind.CORRECTION),
                target_change=EndpointChange(invocation_selector=selector.name),
            )
        )
    return options


def build_blocker_resolutions(
    registry: ModelRegistry,
    config: MigrationRunConfig,
    plan: MigrationPlan,
    analysis: ApplicationAnalysis,
    run_dir: str,
) -> list[BlockerResolution]:
    """One deterministic question-plus-options entry per unresolved blocker."""
    resolutions: list[BlockerResolution] = []
    for blocker in plan.blockers:
        if blocker.category is BlockerCategory.CAPABILITY:
            specific = _capability_options(registry, config, plan, analysis, blocker, run_dir)
        elif blocker.category is BlockerCategory.SOURCE_CONSISTENCY:
            specific = _consistency_options(registry, config, blocker, run_dir)
        elif blocker.category is BlockerCategory.PLATFORM_AMBIGUITY:
            specific = _platform_ambiguity_options(registry, config, blocker, run_dir)
        elif blocker.category is BlockerCategory.CONTEXT_WINDOW:
            specific = _context_window_options(registry, config, analysis, blocker, run_dir)
        elif blocker.category is BlockerCategory.INVALID_SCHEMA:
            specific = _invalid_schema_options(blocker, run_dir)
        elif blocker.category is BlockerCategory.PARAMETER:
            specific = _parameter_options(registry, config, blocker, run_dir)
        elif blocker.category is BlockerCategory.INVOCATION:
            specific = _invocation_options(registry, config, blocker, run_dir)
        else:
            specific = []
        # Invocation blockers offer every reviewed selector: truncating them
        # would hide a legitimate deployment choice.
        cap = len(specific) if blocker.category is BlockerCategory.INVOCATION else 4
        options = [*specific[:cap], _accept_option(run_dir, blocker)]
        no_option_reason = None
        if not specific:
            no_option_reason = (
                "The registry offers no way around this blocker: no other endpoint, model, "
                "or documented redesign resolves it. Fix the issue in the application, or "
                "accept the risk with your reason."
            )
        resolutions.append(
            BlockerResolution(
                blocker=blocker,
                question=_question(blocker, options),
                options=options,
                no_registry_backed_option=no_option_reason,
            )
        )
    return resolutions


# ---------------------------------------------------------------------------
# Decision application (fail closed: no decision, no suppression)
# ---------------------------------------------------------------------------


def _identity_matches(identity: ModelEndpointIdentity, change: EndpointChange) -> bool:
    return all(
        expected is None or actual == expected
        for actual, expected in (
            (identity.model, change.model),
            (identity.platform, change.platform),
            (identity.endpoint, change.endpoint),
        )
    )


def apply_decisions(
    plan: MigrationPlan, log: DecisionLog, config: MigrationRunConfig
) -> MigrationPlan:
    """Re-apply recorded decisions to a freshly generated plan.

    Retarget/correction decisions were already applied to the run's
    migration.yaml when recorded; here they are verified against the current
    config and reported (stale when the identity moved on). Redesign and
    accept decisions suppress exactly the live blocker they name — a decision
    whose blocker no longer exists is reported stale, never applied.
    """
    if not log.decisions:
        return plan
    remaining = {blocker.id: blocker for blocker in plan.blockers}
    required = list(plan.required_changes)
    applied: list[AppliedBlockerDecision] = []
    for index, decision in enumerate(log.decisions):
        status: Literal["applied", "stale", "superseded"]
        if decision.kind in {ResolutionKind.RETARGET, ResolutionKind.CORRECTION}:
            # A correction may fix either side; which change is set says which.
            side = "target" if decision.target_change is not None else "source"
            identity = config.target if side == "target" else config.source
            change = decision.target_change if side == "target" else decision.source_change
            superseded = any(
                later.kind is decision.kind
                and (later.target_change is not None) == (decision.target_change is not None)
                for later in log.decisions[index + 1 :]
            )
            recorded_selector = (
                config.target_invocation_selector
                if side == "target"
                else config.source_invocation_selector
            )
            selector_matches = (
                change is None
                or change.invocation_selector is None
                or recorded_selector == change.invocation_selector
            )
            if change is None or not _identity_matches(identity, change) or not selector_matches:
                if superseded:
                    # This decision was honored when recorded and a newer
                    # decision of the same kind moved the identity onward:
                    # history, not a stale alarm.
                    status = "superseded"
                    detail = (
                        f"A later decision changed the run's {side} again; this "
                        "decision is kept as recorded history."
                    )
                else:
                    status = "stale"
                    detail = (
                        f"The run's {side} identity no longer matches this decision; "
                        "it was not applied."
                    )
            elif decision.blocker_id in remaining:
                status = "stale"
                detail = (
                    "The recorded change is in effect but the same blocker is still "
                    "present; resolve it again from the current options."
                )
            else:
                status = "applied"
                detail = (
                    f"The run's {side} is now {identity.model} on {identity.platform}"
                    f"{_endpoint_suffix(identity.endpoint)}, and the blocker no "
                    "longer occurs in the regenerated plan."
                )
        elif decision.kind is ResolutionKind.REDESIGN_TASK:
            blocker = remaining.get(decision.blocker_id)
            if blocker is None or decision.task is None:
                status = "stale"
                detail = (
                    "No live blocker matches this decision (the application or plan "
                    "changed); the redesign task was not injected."
                )
            else:
                del remaining[decision.blocker_id]
                task = decision.task
                description = task.description
                if task.guidance:
                    description += " Guidance: " + " ".join(task.guidance)
                if task.evidence_urls:
                    description += f" (evidence: {task.evidence_urls[0]})"
                required.append(
                    PlannedMigrationChange(
                        category=task.category,
                        description=description,
                        files=sorted({item.path for item in blocker.locations}),
                        locations=blocker.locations,
                    )
                )
                status = "applied"
                detail = (
                    "The blocker is resolved by redesign; a required "
                    f"{task.category} change with its evidence-linked guidance was "
                    "injected into the plan."
                )
        else:  # ResolutionKind.ACCEPT_WITH_RATIONALE
            if decision.blocker_id in remaining:
                del remaining[decision.blocker_id]
                status = "applied"
                detail = (
                    "The blocker was explicitly accepted with the recorded rationale; "
                    "it is no longer counted as unresolved but stays reported as an "
                    "accepted decision."
                )
            else:
                status = "stale"
                detail = (
                    "No live blocker matches this accepted decision (the application "
                    "or plan changed); nothing was suppressed."
                )
        applied.append(AppliedBlockerDecision(decision=decision, status=status, detail=detail))
    blockers = [blocker for blocker in plan.blockers if blocker.id in remaining]
    applied_count = sum(item.status == "applied" for item in applied)
    accepted_count = sum(
        item.status == "applied" and item.decision.kind is ResolutionKind.ACCEPT_WITH_RATIONALE
        for item in applied
    )
    rationale = [
        *plan.target_selection_rationale,
        (
            f"Recorded user decisions resolve {applied_count} blocker concern(s) "
            f"({accepted_count} accepted as risk); {len(blockers)} blocker(s) remain "
            "unresolved — see the Decisions section for what was traded away and why."
        ),
    ]
    return plan.model_copy(
        update={
            "blockers": blockers,
            "required_changes": required,
            "migration_complexity": migration_complexity(
                blockers, plan.model_differences.highest_severity, required, plan.warnings
            ),
            "target_selection_rationale": rationale,
            "decisions": applied,
        }
    )


def downgrade_accepted_prompt_issues(
    validation: PromptValidationResult, log: DecisionLog
) -> PromptValidationResult:
    """Downgrade validation blockers the user has explicitly accepted.

    The submission gate re-derives prompt validation from scratch; without
    this, an accept decision would clear the plan while the same blocker kept
    rejecting every submission of that prompt — a dead end. Issues are matched
    to decisions through the same id derivation the plan uses, downgraded to
    warnings that name the decision, and recorded on the adaptation entry.
    Only ACCEPT decisions downgrade: a redesign promise does not make an
    unreduced prompt submittable.
    """
    accepted = {
        decision.blocker_id
        for decision in log.decisions
        if decision.kind is ResolutionKind.ACCEPT_WITH_RATIONALE
    }
    if not accepted:
        return validation
    issues = []
    changed = False
    for issue in validation.issues:
        if issue.level is ValidationLevel.BLOCKER and prompt_issue_blocker(issue).id in accepted:
            issues.append(
                issue.model_copy(
                    update={
                        "level": ValidationLevel.WARNING,
                        "message": issue.message
                        + " (explicitly accepted by a recorded blocker decision)",
                    }
                )
            )
            changed = True
        else:
            issues.append(issue)
    if not changed:
        return validation
    return validation.model_copy(
        update={
            "valid": not any(item.level is ValidationLevel.BLOCKER for item in issues),
            "issues": issues,
        }
    )


def render_applied_decision(item: AppliedBlockerDecision) -> str:
    decision = item.decision
    prefix = {"stale": "STALE ", "superseded": "superseded "}.get(item.status, "")
    rationale = (
        f" Rationale: {decision.rationale}"
        if decision.rationale and decision.rationale != decision.summary
        else ""
    )
    return (
        f"{prefix}{decision.kind.value} decision ({decision.decided_on.isoformat()}) "
        f"for blocker {decision.blocker_code} [{decision.blocker_id}]: "
        f"{decision.summary}{rationale} — {item.detail}"
    )
