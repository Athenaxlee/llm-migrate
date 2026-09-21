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

_WORKFLOW_INSTRUCTIONS = """\
llm-migrate plans LLM application migrations locally and deterministically.

For a full migration, prefer the guided workflow over calling low-level tools ad hoc:
1. start_migration(application_path, source, target, ...) — registry-first model
   matching (asks for user confirmation when identifiers are vague), scans the
   application, creates the run workspace (default
   <application>/.llm-migrate/runs/<run-id>/), and reports whether research helps.
2. If research is recommended and the user agrees: get_research_prompts(run_dir),
   run each researcher/reviewer prompt with a separate agent, validate each
   artifact, then build_session_registry(run_dir).
3. list_adaptation_tasks(run_dir) — call it ONCE: the per-file worklist, with
   `shared_prompt_guidance` that applies to every prompt task. Do not re-list
   between submissions; each submission result confirms acceptance.
3b. If the worklist reports blockers: get_blocker_resolutions(run_dir), then
   present each blocker's question with its options, consequences, and
   evidence VERBATIM to the user, ONE blocker at a time, and record each
   answer with record_blocker_decision. The user is the decision maker: never
   pick an option for them, never invent options, and never retry submissions
   to make a blocker disappear. Accepting a blocker requires the user's own
   rationale. Decisions persist in the run's decisions.yaml and re-apply on
   every plan regeneration; each decision result says exactly what to do next.
4. Work through the tasks: adapt prompts minimally (keep wording and structure
   except where a listed model difference or evidence-linked guidance line
   requires a change; each task's `verbatim_source` is the UNMODIFIED original,
   never a proposed adaptation) and submit via submit_adapted_prompt with a
   guidance_dispositions entry for EVERY guidance item (applied /
   not_applicable / declined with a note); write complete adapted files and
   submit via submit_adapted_file with a guidance_dispositions entry for every
   required change, or pass unchanged=true for prompts or files that need no
   change — the report then states explicitly that no change was needed and
   why. Every CHANGED submission must document each edit in annotated_changes:
   exact original/adapted text anchors, a one-sentence why, and evidence
   (kind 'mechanical' for typo-level fixes); undocumented or phantom changes
   are rejected. Submissions are validated and stored under <run>/output/,
   never in the application tree.
5. finalize_migration(run_dir) — writes migration-manifest.yaml and
   migration-report.md (including per-file changes and rationale) under
   output/ and reports any remaining coverage gaps.

Never edit the user's application directly from research results; everything is a
reviewable deliverable in the run's output/ directory.
"""

mcp = FastMCP("llm-migrate", instructions=_WORKFLOW_INSTRUCTIONS)


def _service() -> MigrationService:
    configure_evaluators(os.environ.get("LLM_MIGRATE_EVALUATORS", "").split(","))
    return MigrationService.from_directory(_default_registry())


def _json(model: Any) -> dict[str, Any]:
    return dict(model.model_dump(mode="json"))


@mcp.tool()
def resolve_model(
    identifier: str, platform: str | None = None, endpoint: str | None = None
) -> dict[str, Any]:
    """Match an identifier against the local registry, tolerating vague input.

    Always check here before researching a model: regional Bedrock
    inference-profile prefixes, version suffixes, spacing, and vague platform
    names are normalized deterministically. `status` is `resolved`,
    `needs_confirmation` (show `candidates` to the user and ask), or
    `not_found`.
    """
    match = _service().match_model(identifier, platform, endpoint)
    result = _json(match)
    if match.resolution is not None:
        result["canonical_name"] = match.canonical_name
        # The nested profile is large; fetch it explicitly via get_model_profile.
        result.pop("resolution", None)
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
def scan_application(path: str, prompt_sources: list[str] | None = None) -> dict[str, Any]:
    """Scan a local Python application into a normalized coupling inventory.

    `prompt_sources` optionally names prompt files (relative to the application
    root) that automatic discovery missed; they become explicit high-confidence
    prompt sources.
    """
    return _json(_service().scan_application(path, prompt_sources=prompt_sources))


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


@mcp.tool()
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


@mcp.tool()
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

    Matches both models against the local registry first (tolerating vague or
    platform-decorated identifiers). If either identifier needs confirmation,
    nothing is written and the result carries candidates to show the user.
    Otherwise the run workspace is created (default
    `<application>/.llm-migrate/runs/<run-id>/`, or `output_dir` when given), a
    bounded research request is written only when knowledge is missing or
    stale, and `next_steps` says exactly which tools to call next.
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


@mcp.tool()
def get_research_prompts(run_dir: str) -> dict[str, Any]:
    """Ready-to-run researcher and reviewer prompts for a run's research request.

    Renders one bounded, scope-isolated prompt pair per remaining scope from
    `<run_dir>/request.yaml`. Run each prompt with a separate agent (never let
    one agent research two scopes or review its own research), write the YAML
    artifacts to the stated paths, validate them, then call
    build_session_registry.
    """
    return _json(_service().get_research_prompts(run_dir))


@mcp.tool()
def list_adaptation_tasks(run_dir: str, now: str | None = None) -> dict[str, Any]:
    """Per-file adaptation worklist for a started migration run. Call it once.

    Derived from the run's migration plan (using its session overlay when one
    was built). `shared_prompt_guidance` applies to every prompt task; each
    task carries only its own guidance, and every guidance item has a stable
    id for guidance_dispositions. Each prompt task's `verbatim_source` is the
    ORIGINAL unmodified content — never a proposed adaptation, never to be
    presented as one. Adapt each prompt minimally per the guidance and call
    submit_adapted_prompt (disposing every guidance item); write each complete
    adapted file and call submit_adapted_file (or pass unchanged=true when no
    change is needed). Submission results confirm acceptance, so re-listing
    between submissions is unnecessary; finalize_migration reports remaining
    gaps.
    """
    return _json(
        _service().list_adaptation_tasks(
            run_dir,
            now=datetime.fromisoformat(now) if now else None,
        )
    )


@mcp.tool()
def get_blocker_resolutions(run_dir: str, now: str | None = None) -> dict[str, Any]:
    """Questions plus evidence-backed options for every unresolved blocker.

    For each blocker of a started run, returns the question to ask the user
    and 2-5 registry-backed options (retarget / redesign / correction /
    accept-with-rationale), each with consequences, evidence URLs, and the
    exact record_blocker_decision call to make once the user chooses. Present
    questions, options, and evidence VERBATIM, one blocker at a time; never
    choose on the user's behalf. Also reports how previously recorded
    decisions applied, including stale ones.
    """
    return _json(
        _service().get_blocker_resolutions(
            run_dir,
            now=datetime.fromisoformat(now) if now else None,
        )
    )


@mcp.tool()
def record_blocker_decision(
    run_dir: str,
    blocker_id: str,
    option_id: str,
    rationale: str = "",
    decided_on: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Record the user's decision for one blocker; decisions are durable.

    `blocker_id` and `option_id` must come from get_blocker_resolutions.
    Accept options REQUIRE the user's own free-text `rationale` (they are
    refused without one and are never a default). Retarget and correction
    decisions update the run's migration.yaml identity immediately; redesign
    decisions inject a required, evidence-linked adaptation task on every
    plan regeneration; accept decisions downgrade the blocker to a
    prominently reported accepted decision. The result lists the remaining
    unresolved blockers and the exact next step.
    """
    return _json(
        _service().record_blocker_decision(
            run_dir,
            blocker_id,
            option_id,
            rationale,
            decided_on=date.fromisoformat(decided_on) if decided_on else None,
            now=datetime.fromisoformat(now) if now else None,
        )
    )


@mcp.tool()
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
    """Validate and store one refined prompt for the target model.

    Adapt minimally: keep the original wording and structure except where a
    listed model difference or evidence-linked guidance item requires a
    change. `guidance_dispositions` must dispose EVERY guidance item of this
    prompt's task (its `guidance` plus the worklist's
    `shared_prompt_guidance`) exactly once, each entry `{"guidance_id": ...,
    "disposition": "applied" | "not_applicable" | "declined", "note": ...}`
    (a decline requires the note); missing or unknown ids are rejected.
    Every edit must be documented in `annotated_changes`: each entry
    `{"operation": "edit" | "insert" | "delete" | "restructure",
    "original_anchor": <exact span of the original decoded content, for
    edit/delete>, "adapted_anchor": <exact span of the adapted content, for
    edit/insert>, "why": <one sentence: which target-model behavior needs
    it>, "evidence": [{"kind": "model_guidance" | "model_difference" |
    "analysis_finding" | "research" | "mechanical", "url": ..., "reference":
    ...}]}`. Anchors are matched against the DECODED runtime values; every
    diff hunk must be covered and every annotation must match a real edit
    (undocumented or phantom changes are rejected); non-mechanical evidence
    needs a url or reference. Validation runs on the DECODED runtime prompt
    values, and a submission whose decoded values equal the original's — or
    differ only by whitespace or letter case — is rejected: serialization
    tricks, escapes, quoting and cosmetic edits are never an adaptation. If
    the prompt needs no change, pass `unchanged=true` with an EMPTY
    `adapted_prompt` (combining it with content is an error) to record a
    reviewed no-change deliverable — the final report then states explicitly
    that no change was needed and why; the claim is refused when the prompt's
    decoded values still reference the source model, and it cannot mark any
    guidance item as applied or carry annotated changes. Blockers are
    rejected, and a submission that drops the original's structural sections
    (XML-like tags or prompt components) is rejected unless
    `allow_restructure` is true and the justification is recorded (a
    `restructure` annotation with no anchors claims the whole rewrite).
    Accepted prompts are written beneath `<run>/output/prompts/` mirroring
    the application layout, and the annotations/dispositions appear verbatim
    in the final migration report.
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


@mcp.tool()
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
    """Store one finalized post-adaptation application file for review.

    `adapted_content` must be the complete file, not a diff.
    `guidance_dispositions` must dispose every required change of this
    file's task exactly once (same shape and rules as prompt submissions),
    and every edit to an existing file must be documented in
    `annotated_changes` (same shape and rules as prompt submissions, with
    anchors matched against the raw file content). Submissions are checked
    deterministically (path containment, Python syntax, model-id
    consistency, annotation-diff reconciliation) and written beneath
    `<run>/output/files/`; the application tree itself is never modified. If
    the file genuinely needs no change for the target model, pass
    `unchanged=true` with an empty `adapted_content`: the original is copied
    as the deliverable, the coverage gap closes without resending the file,
    and the final report states explicitly that no change was needed and
    why. Prompt source files are rejected here; submit them with
    submit_adapted_prompt instead.
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


@mcp.tool()
def finalize_migration(run_dir: str, now: str | None = None) -> dict[str, Any]:
    """Write the run's manifest, report, and adaptation change log under output/.

    The report includes, per adapted file, what changed and why; deliverables
    reviewed with no change needed are stated explicitly with their reasoning
    (and counted separately as `reviewed_unchanged`); plus the affected files
    that still lack an adaptation deliverable. Re-run it any time; it always
    reflects the current submissions.
    """
    return _json(
        _service().finalize_migration_run(
            run_dir,
            now=datetime.fromisoformat(now) if now else None,
        )
    )


def main() -> None:
    """Run the MCP server using FastMCP's default stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
