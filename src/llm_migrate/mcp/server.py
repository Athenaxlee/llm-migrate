"""Thin FastMCP wrappers around :class:`MigrationService`."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from fastmcp import FastMCP

from llm_migrate.cli.main import _default_registry
from llm_migrate.core.agent_research import (
    EvidenceReview,
    MigrationResearchRequest,
    ModelEndpointIdentity,
    ResearchTopic,
)
from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRunConfig,
    EvaluationSuite,
    MigrationEvalRun,
    OptimizationLimits,
    RegressionReport,
)
from llm_migrate.core.evaluator_registry import configure_evaluators
from llm_migrate.core.knowledge import ResearchResult
from llm_migrate.core.models import MigrationPlan, MigrationWorkload, RecommendationConstraints
from llm_migrate.core.moments import utc_moment
from llm_migrate.core.orchestration import NullAgentRunner
from llm_migrate.core.session import manifest_summary_lines
from llm_migrate.service import MigrationService

_WORKFLOW_INSTRUCTIONS = """\
llm-migrate plans LLM application migrations locally and deterministically.

For a full migration, use the guided workflow (LLM_MIGRATE_TOOLSET=guided
exposes only its tools):
1. start_migration(application_path, source, target, ...) — registry-first
   matching (asks for user confirmation when identifiers are vague), scans the
   application, creates the run workspace, and reports whether research helps.
2. If research is recommended and the user agrees: get_research_prompts(run_dir),
   run each researcher/reviewer prompt with a separate agent, validate each
   artifact with validate_research_artifact(run_dir, scope), then
   build_session_registry(run_dir).
3. list_adaptation_tasks(run_dir) — call it ONCE; `shared_prompt_guidance`
   applies to every prompt task, and submission results confirm acceptance, so
   never re-list between submissions.
3b. If the worklist reports blockers: get_blocker_resolutions(run_dir); present
   each blocker's question, options, consequences, and evidence VERBATIM, one
   blocker at a time, and record each user answer with record_blocker_decision.
   The user decides: never pick for them, never invent options, never retry
   submissions to clear a blocker; accepts require the user's own rationale.
4. Work the tasks: adapt each prompt minimally (its `verbatim_source` is the
   UNMODIFIED original, never a proposed adaptation) and submit_adapted_prompt;
   write complete adapted files and submit_adapted_file; pass unchanged=true
   when no change is needed. Every submission disposes every guidance item in
   guidance_dispositions and documents every edit in annotated_changes
   (anchors, why, evidence); undocumented or phantom changes are rejected.
   Deliverables live under <run>/output/, never in the application tree.
5. finalize_migration(run_dir) — writes the manifest and report and lists
   remaining coverage gaps and undecided changes.
6. get_change_review(run_dir) lists every annotated change with its decision
   state; present each pending change VERBATIM, one at a time, and record the
   user's accept or reject with record_change_decision (a rejection
   deterministically regenerates the deliverable; a resubmission makes
   decisions stale — reported, never silently applied).

Never edit the user's application directly; everything is a reviewable
deliverable in the run's output/ directory.
"""

mcp = FastMCP("llm-migrate", instructions=_WORKFLOW_INSTRUCTIONS)

# The guided workflow needs only these tools; LLM_MIGRATE_TOOLSET=guided keeps
# the exposed tool surface within host inline-tool budgets by hiding the rest.
_GUIDED_TOOL_NAMES = frozenset(
    {
        "resolve_model",
        "get_model_profile",
        "start_migration",
        "get_research_prompts",
        "validate_research_artifact",
        "build_session_registry",
        "list_adaptation_tasks",
        "get_blocker_resolutions",
        "record_blocker_decision",
        "submit_adapted_prompt",
        "submit_adapted_file",
        "finalize_migration",
        "get_change_review",
        "record_change_decision",
    }
)

_TOOLSET = os.environ.get("LLM_MIGRATE_TOOLSET", "full").strip().casefold() or "full"


def _tool(fn: Callable[..., Any]) -> Any:
    """Register one MCP tool, honoring the LLM_MIGRATE_TOOLSET selection.

    With `LLM_MIGRATE_TOOLSET=guided` only the guided-workflow tools are
    exposed over MCP; any other value (default `full`) exposes everything.
    The Python function stays importable and callable either way.
    """
    if _TOOLSET == "guided" and fn.__name__ not in _GUIDED_TOOL_NAMES:
        return fn
    return mcp.tool()(fn)


def _service() -> MigrationService:
    configure_evaluators(os.environ.get("LLM_MIGRATE_EVALUATORS", "").split(","))
    return MigrationService.from_directory(_default_registry())


def _json(model: Any) -> dict[str, Any]:
    return dict(model.model_dump(mode="json"))


@_tool
def resolve_model(
    identifier: str, platform: str | None = None, endpoint: str | None = None
) -> dict[str, Any]:
    """Match an identifier against the local registry, tolerating vague input.

    Check here before researching a model: prefixes, version suffixes, and
    vague platform names normalize deterministically. `status` is `resolved`,
    `needs_confirmation` (show `candidates` to the user), or `not_found`.
    """
    match = _service().match_model(identifier, platform, endpoint)
    result = _json(match)
    if match.resolution is not None:
        result["canonical_name"] = match.canonical_name
        # The nested profile is large; fetch it explicitly via get_model_profile.
        result.pop("resolution", None)
    return result


@_tool
def get_model_profile(identifier: str) -> dict[str, Any]:
    """Return a validated local registry model profile."""
    return _json(_service().get_model_profile(identifier))


@_tool
def check_model_lifecycle(
    identifier: str,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Interpret reviewed lifecycle facts at a fixed date without live research."""
    effective_date = date.fromisoformat(as_of_date) if as_of_date else date.today()
    return _json(_service().check_model_lifecycle(identifier, as_of_date=effective_date))


@_tool
def compare_models(
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
    source_endpoint: str | None = None,
    target_endpoint: str | None = None,
    include_live_pricing: bool = False,
) -> dict[str, Any]:
    """Return explicit same/different/unsupported/unknown comparison states."""
    return _json(
        _service().compare_models(
            source,
            target,
            source_platform,
            target_platform,
            source_endpoint,
            target_endpoint,
            include_live_pricing=include_live_pricing,
        )
    )


@_tool
def recommend_models(
    platform: str | None = None,
    provider: str | None = None,
    region: str | None = None,
    required_capabilities: list[str] | None = None,
    minimum_context_window: int | None = None,
    migration_goal: str = "balanced",
    source_model: str | None = None,
    source_platform: str | None = None,
    application_path: str | None = None,
    include_live_pricing: bool = False,
) -> dict[str, Any]:
    """Hard-filter and deterministically rank compatible registry models."""
    constraints = RecommendationConstraints.model_validate(
        {
            "platform": platform,
            "provider": provider,
            "region": region,
            "required_capabilities": required_capabilities or [],
            "minimum_context_window": minimum_context_window,
            "migration_goal": migration_goal,
            "source_model": source_model,
            "source_platform": source_platform,
        }
    )
    service = _service()
    analysis = service.scan_application(application_path) if application_path else None
    return _json(
        service.recommend_models(constraints, analysis, include_live_pricing=include_live_pricing)
    )


@_tool
def scan_application(path: str, prompt_sources: list[str] | None = None) -> dict[str, Any]:
    """Scan a local Python application into a normalized coupling inventory.

    `prompt_sources` names prompt files automatic discovery missed (relative
    to the application root).
    """
    return _json(_service().scan_application(path, prompt_sources=prompt_sources))


@_tool
def query_live_pricing(identifier: str, timeout: float = 5.0) -> dict[str, Any]:
    """Opt in to an OpenRouter pricing inquiry; canonical registry facts are unchanged."""
    return _json(_service().query_live_pricing(identifier, timeout=timeout))


@_tool
def estimate_migration_cost(
    source: str,
    target: str,
    requests: int,
    input_tokens_per_request: int,
    output_tokens_per_request: int,
) -> dict[str, Any]:
    """Estimate recurring token cost from checked-in canonical pricing."""
    workload = MigrationWorkload(
        requests=requests,
        input_tokens_per_request=input_tokens_per_request,
        output_tokens_per_request=output_tokens_per_request,
    )
    return _json(_service().estimate_migration_cost(source, target, workload))


@_tool
def analyze_prompt(prompt: str) -> dict[str, Any]:
    """Conservatively analyze prompt characteristics without inference."""
    return _json(_service().analyze_prompt(prompt))


@_tool
def prepare_prompt_migration(
    source: str,
    target: str,
    prompt: str,
    source_path: str | None = None,
    source_role: Literal["system", "user", "developer", "unknown"] = "unknown",
    source_platform: str | None = None,
    target_platform: str | None = None,
) -> dict[str, Any]:
    """Prepare a prompt migration specification without rewriting the prompt."""
    return _json(
        _service().prepare_prompt_migration(
            source,
            target,
            prompt,
            source_path=source_path,
            source_role=source_role,
            source_platform=source_platform,
            target_platform=target_platform,
        )
    )


@_tool
def validate_prompt(
    target: str,
    prompt: str,
    source_path: str | None = None,
    target_platform: str | None = None,
    target_endpoint: str | None = None,
) -> dict[str, Any]:
    """Statically validate prompt assumptions against target registry facts."""
    return _json(
        _service().validate_prompt(
            target,
            prompt,
            source_path=source_path,
            target_platform=target_platform,
            target_endpoint=target_endpoint,
        )
    )


@_tool
def analyze_invocation(
    application_path: str,
    target: str | None = None,
    target_platform: str | None = None,
) -> dict[str, Any]:
    """Analyze provider invocation and adjacent tool/output couplings."""
    return _json(
        _service().analyze_invocation(
            application_path, target=target, target_platform=target_platform
        )
    )


@_tool
def prepare_invocation_migration(
    application_path: str,
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
) -> dict[str, Any]:
    """Prepare a target invocation contract without editing application files."""
    return _json(
        _service().prepare_invocation_migration(
            application_path,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
        )
    )


@_tool
def generate_migration_plan(
    application_path: str,
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
    source_endpoint: str | None = None,
    target_endpoint: str | None = None,
    prompt_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Generate an actionable application-level migration manifest without writing files."""
    return _json(
        _service().generate_migration_plan(
            application_path,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            prompt_sources=prompt_sources,
        )
    )


@_tool
def generate_migration_report(
    application_path: str,
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
    source_endpoint: str | None = None,
    target_endpoint: str | None = None,
    prompt_sources: list[str] | None = None,
) -> str:
    """Generate a human-readable report from the integrated migration workflow."""
    return _service().generate_migration_report(
        application_path,
        source,
        target,
        source_platform=source_platform,
        target_platform=target_platform,
        source_endpoint=source_endpoint,
        target_endpoint=target_endpoint,
        prompt_sources=prompt_sources,
    )


@_tool
def generate_eval_suite(
    migration_plan: dict[str, Any],
    cases: list[dict[str, Any]],
    name: str = "migration-evaluation",
) -> dict[str, Any]:
    """Bind a deterministic evaluation corpus to a migration manifest."""
    plan = MigrationPlan.model_validate(migration_plan)
    corpus = [EvaluationCase.model_validate(case) for case in cases]
    return _json(_service().generate_eval_suite(plan, corpus, name=name))


@_tool
def run_migration_eval(
    suite: dict[str, Any],
    source_config: dict[str, Any],
    target_config: dict[str, Any],
) -> dict[str, Any]:
    """Run source and target with user-owned local provider credentials."""
    return _json(
        _service().run_migration_eval(
            EvaluationSuite.model_validate(suite),
            EvaluationRunConfig.model_validate(source_config),
            EvaluationRunConfig.model_validate(target_config),
        )
    )


@_tool
def compare_outputs(source_result: dict[str, Any], target_result: dict[str, Any]) -> dict[str, Any]:
    """Compare one paired source/target evaluation result deterministically."""
    return _json(
        _service().compare_outputs(
            EvaluationCaseResult.model_validate(source_result),
            EvaluationCaseResult.model_validate(target_result),
        )
    )


@_tool
def analyze_regressions(run: dict[str, Any]) -> dict[str, Any]:
    """Return a structured categorized regression report for an evaluation run."""
    return _json(_service().analyze_regressions(MigrationEvalRun.model_validate(run)))


@_tool
def optimize_migration(
    regression_report: dict[str, Any],
    candidate_runs: dict[str, dict[str, Any]] | None = None,
    limits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return bounded, review-only recommendations from observed regression evidence."""
    runs = {
        name: MigrationEvalRun.model_validate(run) for name, run in (candidate_runs or {}).items()
    }
    return _json(
        _service().optimize_migration(
            RegressionReport.model_validate(regression_report),
            runs,
            limits=OptimizationLimits.model_validate(limits) if limits is not None else None,
        )
    )


@_tool
def propose_registry_update(research: dict[str, Any]) -> dict[str, Any]:
    """Plan a registry update from supplied evidence without editing canonical files."""
    result = ResearchResult.model_validate(research)
    return _json(_service().propose_registry_update(result))


@_tool
def create_migration_research_request(
    application: str,
    run_id: str,
    source: dict[str, Any],
    target: dict[str, Any],
    as_of: str,
    topics: list[str] | None = None,
) -> dict[str, Any]:
    """Scan locally and bound explicit user-scoped research to missing/stale facts."""
    request = _service().create_migration_research_request(
        Path(application),
        run_id=run_id,
        source=ModelEndpointIdentity.model_validate(source),
        target=ModelEndpointIdentity.model_validate(target),
        as_of=date.fromisoformat(as_of),
        topics=[ResearchTopic(topic) for topic in topics] if topics else None,
    )
    return _json(request)


@_tool
def validate_research_result(
    research: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Run the deterministic scope/policy gate over one research artifact."""
    problems = _service().validate_research_result(
        ResearchResult.model_validate(research),
        MigrationResearchRequest.model_validate(request),
    )
    return {"valid": not problems, "problems": problems}


@_tool
def validate_evidence_review(
    review: dict[str, Any],
    research: dict[str, Any],
) -> dict[str, Any]:
    """Check that an independent evidence review actually covers the research artifact."""
    problems = _service().validate_evidence_review(
        EvidenceReview.model_validate(review),
        ResearchResult.model_validate(research),
    )
    return {"valid": not problems, "problems": problems}


@_tool
def validate_research_artifact(run_dir: str, scope: str) -> dict[str, Any]:
    """Validate one scope's research artifacts directly from the run workspace.

    Reads `request.yaml`, `research/<scope>.yaml`, and (when present)
    `review/<scope>.yaml` and runs the deterministic scope/policy and
    review-integrity gates; nothing is resent through the payload. `scope`
    is `source`, `target`, or `pair`.
    """
    return _json(_service().validate_research_artifact(run_dir, scope))


@_tool
def build_research_consensus(
    research: dict[str, Any],
    review: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically combine research and review; agent agreement is not evidence."""
    return _json(
        _service().build_research_consensus(
            ResearchResult.model_validate(research),
            EvidenceReview.model_validate(review),
            MigrationResearchRequest.model_validate(request),
        )
    )


@_tool
def build_session_registry(
    run_dir: str,
    now: str | None = None,
    ttl_days: int = 7,
    shadow_canonical: list[str] | None = None,
) -> dict[str, Any]:
    """Finalize a run workspace into an immutable, expiring session overlay.

    Research and review artifacts must already exist in the workspace; any
    stage still needing an agent fails closed instead of doing hidden work.
    """
    workspace = Path(run_dir)
    request = MigrationResearchRequest.model_validate(
        yaml.safe_load((workspace / "request.yaml").read_text(encoding="utf-8"))
    )
    moment = utc_moment(now) or datetime.now(UTC)
    outcome = _service().run_agent_research(
        request,
        NullAgentRunner(),
        workspace.parent,
        now=moment,
        overlay_ttl=timedelta(days=ttl_days),
        shadow_canonical=shadow_canonical,
    )
    return _json(outcome)


@_tool
def generate_session_migration_plan(
    application: str,
    source: str,
    target: str,
    session_run_dir: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Plan a migration over the canonical registry plus one run's session overlay."""
    base = _service()
    session_service, manifest = base.load_session_service(
        Path(session_run_dir),
        as_of=utc_moment(now) or datetime.now(UTC),
    )
    plan = session_service.generate_migration_plan(
        Path(application),
        source,
        target,
        source_platform=source_platform,
        target_platform=target_platform,
    )
    plan = plan.model_copy(update={"warnings": [*plan.warnings, *manifest_summary_lines(manifest)]})
    return _json(plan)


@_tool
def start_migration(
    application_path: str,
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
    source_endpoint: str | None = None,
    target_endpoint: str | None = None,
    run_id: str | None = None,
    output_dir: str | None = None,
    as_of: str | None = None,
    research: Literal["auto", "skip"] = "auto",
    prompt_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Start a guided migration run; the preferred entry point for a full migration.

    Both models match registry-first; if either needs confirmation, nothing
    is written and `candidates` must be shown to the user. Otherwise the run
    workspace is created, a bounded research request is written only when
    knowledge is missing or stale, and `next_steps` says exactly what to
    call next.
    """
    start = _service().start_migration_run(
        Path(application_path),
        source,
        target,
        source_platform=source_platform,
        target_platform=target_platform,
        source_endpoint=source_endpoint,
        target_endpoint=target_endpoint,
        run_id=run_id,
        output_dir=Path(output_dir) if output_dir else None,
        as_of=date.fromisoformat(as_of) if as_of else None,
        research=research,
        prompt_sources=prompt_sources,
    )
    result = _json(start)
    # The nested profiles are large; fetch one explicitly via get_model_profile.
    for side in ("source_match", "target_match"):
        result[side].pop("resolution", None)
    return result


@_tool
def get_research_prompts(run_dir: str) -> dict[str, Any]:
    """Ready-to-run researcher and reviewer prompts for a run's research request.

    One bounded prompt pair per remaining scope. Run each with a separate
    agent (never one agent for two scopes or reviewing its own research),
    write the YAML artifacts to the stated paths, validate each with
    validate_research_artifact, then call build_session_registry.
    """
    return _json(_service().get_research_prompts(run_dir))


@_tool
def list_adaptation_tasks(run_dir: str, now: str | None = None) -> dict[str, Any]:
    """Per-file adaptation worklist for a started migration run. Call it ONCE.

    `shared_prompt_guidance` applies to every prompt task; guidance items
    carry stable ids for guidance_dispositions. Each prompt task's
    `verbatim_source` is the ORIGINAL unmodified content, never a proposed
    adaptation. Do not re-list between submissions; finalize_migration
    reports remaining gaps.
    """
    return _json(
        _service().list_adaptation_tasks(
            run_dir,
            now=utc_moment(now),
        )
    )


@_tool
def get_blocker_resolutions(run_dir: str, now: str | None = None) -> dict[str, Any]:
    """Questions plus evidence-backed options for every unresolved blocker.

    Each blocker carries the question to ask the user and registry-backed
    options with consequences, evidence URLs, and the exact
    record_blocker_decision call. Present them VERBATIM, one blocker at a
    time; never choose on the user's behalf.
    """
    return _json(
        _service().get_blocker_resolutions(
            run_dir,
            now=utc_moment(now),
        )
    )


@_tool
def record_blocker_decision(
    run_dir: str,
    blocker_id: str,
    option_id: str,
    rationale: str = "",
    decided_on: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Record the user's decision for one blocker; decisions are durable.

    Ids must come from get_blocker_resolutions. Accept options REQUIRE the
    user's own free-text rationale and are never a default.
    Retarget/correction decisions update the run identity immediately;
    redesign decisions inject a required task on every regeneration; accept
    decisions downgrade the blocker to a reported accepted risk. The result
    lists the remaining blockers and the exact next step.
    """
    return _json(
        _service().record_blocker_decision(
            run_dir,
            blocker_id,
            option_id,
            rationale,
            decided_on=date.fromisoformat(decided_on) if decided_on else None,
            now=utc_moment(now),
        )
    )


@_tool
def submit_adapted_prompt(
    run_dir: str,
    source_path: str,
    adapted_prompt: str,
    rationale: str,
    changes: list[str] | None = None,
    allow_restructure: bool = False,
    unchanged: bool = False,
    guidance_dispositions: list[dict[str, str]] | None = None,
    annotated_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate and store one adapted prompt for the target model.

    Adapt minimally: change only what a listed model difference or guidance
    item requires. Dispose EVERY guidance item of the task (its `guidance`
    plus `shared_prompt_guidance`) exactly once in `guidance_dispositions`:
    {guidance_id, disposition: applied|not_applicable|declined, note
    (required when declined)}. Document every edit in `annotated_changes`
    against the DECODED runtime values: {operation: edit|insert|delete|
    restructure, original_anchor/adapted_anchor (exact spans), why,
    evidence: [{kind: model_guidance|model_difference|analysis_finding|
    research|mechanical, url|reference}]}; every diff hunk covered, no
    phantom annotations. Serialization-, whitespace-, or case-only edits
    are rejected; structural drops need allow_restructure=true plus a
    recorded justification. No change needed? Pass unchanged=true with an
    EMPTY adapted_prompt (refused while the prompt still references the
    source model). Deliverables land under <run>/output/prompts/.
    """
    return _json(
        _service().submit_adapted_prompt(
            run_dir,
            source_path,
            adapted_prompt,
            rationale,
            changes,
            allow_restructure=allow_restructure,
            unchanged=unchanged,
            guidance_dispositions=guidance_dispositions,
            annotated_changes=annotated_changes,
        )
    )


@_tool
def submit_adapted_file(
    run_dir: str,
    source_path: str,
    adapted_content: str,
    rationale: str,
    changes: list[str] | None = None,
    new_file: bool = False,
    unchanged: bool = False,
    guidance_dispositions: list[dict[str, str]] | None = None,
    annotated_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Store one complete adapted application file (never a diff) for review.

    Same rules as submit_adapted_prompt: dispose every required change in
    `guidance_dispositions` and document every edit in `annotated_changes`
    (anchors match the raw file content). Deterministic checks gate the
    write to `<run>/output/files/`; the application tree is never modified.
    Pass unchanged=true with empty adapted_content for a reviewed no-change
    file. Prompt sources are rejected here; use submit_adapted_prompt.
    """
    return _json(
        _service().submit_adapted_file(
            run_dir,
            source_path,
            adapted_content,
            rationale,
            changes,
            new_file=new_file,
            unchanged=unchanged,
            guidance_dispositions=guidance_dispositions,
            annotated_changes=annotated_changes,
        )
    )


@_tool
def finalize_migration(run_dir: str, now: str | None = None) -> dict[str, Any]:
    """Write the run's manifest, report, and adaptation change log under output/.

    The report records per-file changes and rationale, states reviewed
    no-change deliverables explicitly, and lists remaining coverage gaps.
    Re-run it any time; it always reflects the current submissions.
    """
    return _json(
        _service().finalize_migration_run(
            run_dir,
            now=utc_moment(now),
        )
    )


@_tool
def get_change_review(run_dir: str) -> dict[str, Any]:
    """Per-deliverable annotated changes paired with their decision state.

    Returns each deliverable's decoded diff and every annotated change (why,
    evidence, spans, pending/accepted/rejected). Present each pending change
    VERBATIM, one at a time; the user decides, never you. Drift and stale
    decisions are reported, never silently applied.
    """
    return _json(_service().get_change_review(run_dir))


@_tool
def record_change_decision(
    run_dir: str,
    source_path: str,
    change_id: str,
    decision: str,
    note: str = "",
    decided_on: str | None = None,
) -> dict[str, Any]:
    """Record the user's accept/reject for one annotated change; durable.

    `change_id` comes from get_change_review. Every decision
    deterministically regenerates the deliverable from the original, the
    as-submitted content, and all live rejections; if that cannot be done
    deterministically the decision is refused (NOT recorded) with the
    reason. The result lists the change ids still pending for the file.
    """
    return _json(
        _service().record_change_decision(
            run_dir,
            source_path,
            change_id,
            decision,
            note,
            decided_on=date.fromisoformat(decided_on) if decided_on else None,
        )
    )


def main() -> None:
    """Run the MCP server using FastMCP's default stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
