"""Thin FastMCP wrappers around :class:`MigrationService`."""

from __future__ import annotations

import os
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
from llm_migrate.core.orchestration import NullAgentRunner
from llm_migrate.core.session import manifest_summary_lines
from llm_migrate.service import MigrationService

mcp = FastMCP("llm-migrate")


def _service() -> MigrationService:
    configure_evaluators(os.environ.get("LLM_MIGRATE_EVALUATORS", "").split(","))
    return MigrationService.from_directory(_default_registry())


def _json(model: Any) -> dict[str, Any]:
    return dict(model.model_dump(mode="json"))


@mcp.tool()
def resolve_model(
    identifier: str, platform: str | None = None, endpoint: str | None = None
) -> dict[str, Any]:
    """Resolve a registry identifier with optional platform context."""
    resolution = _service().resolve_model(identifier, platform, endpoint)
    result = _json(resolution)
    result["canonical_name"] = resolution.canonical_name
    return result


@mcp.tool()
def get_model_profile(identifier: str) -> dict[str, Any]:
    """Return a validated local registry model profile."""
    return _json(_service().get_model_profile(identifier))


@mcp.tool()
def check_model_lifecycle(
    identifier: str,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """Interpret reviewed lifecycle facts at a fixed date without live research."""
    effective_date = date.fromisoformat(as_of_date) if as_of_date else date.today()
    return _json(_service().check_model_lifecycle(identifier, as_of_date=effective_date))


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def scan_application(path: str) -> dict[str, Any]:
    """Scan a local Python application into a normalized coupling inventory."""
    return _json(_service().scan_application(path))


@mcp.tool()
def query_live_pricing(identifier: str, timeout: float = 5.0) -> dict[str, Any]:
    """Opt in to an OpenRouter pricing inquiry; canonical registry facts are unchanged."""
    return _json(_service().query_live_pricing(identifier, timeout=timeout))


@mcp.tool()
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


@mcp.tool()
def analyze_prompt(prompt: str) -> dict[str, Any]:
    """Conservatively analyze prompt characteristics without inference."""
    return _json(_service().analyze_prompt(prompt))


@mcp.tool()
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


@mcp.tool()
def validate_prompt(
    target: str,
    prompt: str,
    source_path: str | None = None,
    target_platform: str | None = None,
) -> dict[str, Any]:
    """Statically validate prompt assumptions against target registry facts."""
    return _json(
        _service().validate_prompt(
            target,
            prompt,
            source_path=source_path,
            target_platform=target_platform,
        )
    )


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def generate_migration_plan(
    application_path: str,
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
) -> dict[str, Any]:
    """Generate an actionable application-level migration manifest without writing files."""
    return _json(
        _service().generate_migration_plan(
            application_path,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
        )
    )


@mcp.tool()
def generate_migration_report(
    application_path: str,
    source: str,
    target: str,
    source_platform: str | None = None,
    target_platform: str | None = None,
) -> str:
    """Generate a human-readable report from the integrated migration workflow."""
    return _service().generate_migration_report(
        application_path,
        source,
        target,
        source_platform=source_platform,
        target_platform=target_platform,
    )


@mcp.tool()
def generate_eval_suite(
    migration_plan: dict[str, Any],
    cases: list[dict[str, Any]],
    name: str = "migration-evaluation",
) -> dict[str, Any]:
    """Bind a deterministic evaluation corpus to a migration manifest."""
    plan = MigrationPlan.model_validate(migration_plan)
    corpus = [EvaluationCase.model_validate(case) for case in cases]
    return _json(_service().generate_eval_suite(plan, corpus, name=name))


@mcp.tool()
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


@mcp.tool()
def compare_outputs(source_result: dict[str, Any], target_result: dict[str, Any]) -> dict[str, Any]:
    """Compare one paired source/target evaluation result deterministically."""
    return _json(
        _service().compare_outputs(
            EvaluationCaseResult.model_validate(source_result),
            EvaluationCaseResult.model_validate(target_result),
        )
    )


@mcp.tool()
def analyze_regressions(run: dict[str, Any]) -> dict[str, Any]:
    """Return a structured categorized regression report for an evaluation run."""
    return _json(_service().analyze_regressions(MigrationEvalRun.model_validate(run)))


@mcp.tool()
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


@mcp.tool()
def propose_registry_update(research: dict[str, Any]) -> dict[str, Any]:
    """Plan a registry update from supplied evidence without editing canonical files."""
    result = ResearchResult.model_validate(research)
    return _json(_service().propose_registry_update(result))


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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
    moment = datetime.fromisoformat(now) if now else datetime.now(UTC)
    outcome = _service().run_agent_research(
        request,
        NullAgentRunner(),
        workspace.parent,
        now=moment,
        overlay_ttl=timedelta(days=ttl_days),
        shadow_canonical=shadow_canonical,
    )
    return _json(outcome)


@mcp.tool()
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
        as_of=datetime.fromisoformat(now) if now else datetime.now(UTC),
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


def main() -> None:
    """Run the MCP server using FastMCP's default stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
