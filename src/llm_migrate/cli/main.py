"""Typer command-line interface for llm-migrate."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
import yaml
from pydantic import ValidationError

from llm_migrate.core.agent_research import (
    EvidenceReview,
    MigrationResearchRequest,
    ModelEndpointIdentity,
    ResearchTopic,
)
from llm_migrate.core.artifacts import proposal_as_markdown, proposal_as_yaml
from llm_migrate.core.evaluation_models import (
    EvaluationCase,
    EvaluationRunConfig,
    EvaluationSuite,
    MigrationEvalRun,
    OptimizationLimits,
    RegressionReport,
)
from llm_migrate.core.evaluator_registry import configure_evaluators
from llm_migrate.core.knowledge import ResearchResult
from llm_migrate.core.models import (
    MigrationGoal,
    MigrationPlan,
    MigrationWorkload,
    RecommendationConstraints,
)
from llm_migrate.core.moments import utc_moment
from llm_migrate.core.registry import RegistryError
from llm_migrate.core.workspace import GuidanceDisposition
from llm_migrate.service import MigrationService

app = typer.Typer(help="Plan local, provider-neutral LLM application migrations.")
models_app = typer.Typer(help="Inspect and resolve registry models.")
prompt_app = typer.Typer(help="Analyze prompts and prepare prompt migrations.")
invocation_app = typer.Typer(help="Analyze and prepare provider invocation migrations.")
registry_app = typer.Typer(help="Validate and propose changes to local registry knowledge.")
eval_app = typer.Typer(help="Run source/target evaluations and analyze regressions.")
research_app = typer.Typer(
    help="Stage operations for explicitly requested, user-scoped agent research."
)
run_app = typer.Typer(
    help="Guided migration runs with a workspace, adaptation deliverables, and reports."
)
app.add_typer(models_app, name="models")
app.add_typer(prompt_app, name="prompt")
app.add_typer(invocation_app, name="invocation")
app.add_typer(registry_app, name="registry")
app.add_typer(eval_app, name="eval")
app.add_typer(research_app, name="research")
app.add_typer(run_app, name="run")


def _default_registry() -> Path:
    configured = os.environ.get("LLM_MIGRATE_REGISTRY")
    if configured:
        return Path(configured)
    candidate = Path.cwd() / "registry"
    if candidate.is_dir():
        return candidate
    source_candidate = Path(__file__).resolve().parents[3] / "registry"
    if source_candidate.is_dir():
        return source_candidate
    return Path(sys.prefix) / "share" / "llm-migrate" / "registry"


def _service(registry: Path | None = None) -> MigrationService:
    try:
        configured_evaluators = os.environ.get("LLM_MIGRATE_EVALUATORS", "")
        configure_evaluators(configured_evaluators.split(","))
        return MigrationService.from_directory(registry or _default_registry())
    except RegistryError as exc:
        typer.echo(f"Registry error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ValueError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _emit(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, list):
        value = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in value
        ]
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _apply_session_overlay(
    service: MigrationService,
    session: Path | None,
    session_as_of: str | None = None,
) -> tuple[MigrationService, list[str]]:
    """Swap in a hash-verified session overlay and surface its provenance."""
    if session is None:
        return service, []
    from llm_migrate.core.session import manifest_summary_lines

    moment = utc_moment(session_as_of) or datetime.now(UTC)
    try:
        overlay_service, manifest = service.load_session_service(session, as_of=moment)
    except (RegistryError, ValueError) as exc:
        typer.echo(f"Unable to load session overlay: {exc}", err=True)
        raise typer.Exit(2) from exc
    return overlay_service, manifest_summary_lines(manifest)


def _read_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Unable to read prompt {path}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _read_research(path: Path) -> ResearchResult:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ResearchResult.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        typer.echo(f"Invalid research result {path}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _write_migration_artifact(
    output: Path,
    rendered: str,
    application: Path,
    affected_files: list[str],
    label: str,
) -> None:
    base = application if application.is_dir() else application.parent
    protected = {(base / path).resolve() for path in affected_files}
    if output.resolve() in protected:
        typer.echo(
            f"Refusing to overwrite analyzed source file {output} with a {label}.",
            err=True,
        )
        raise typer.Exit(2)
    try:
        output.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Unable to write {label} {output}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _read_yaml(path: Path, label: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        typer.echo(f"Invalid {label} {path}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _write_eval_artifact(output: Path, rendered: str, inputs: list[Path], label: str) -> None:
    if output.resolve() in {path.resolve() for path in inputs}:
        typer.echo(f"Refusing to overwrite an evaluation input with the {label}.", err=True)
        raise typer.Exit(2)
    try:
        output.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        typer.echo(f"Unable to write {label} {output}: {exc}", err=True)
        raise typer.Exit(2) from exc


@registry_app.command("validate")
def registry_validate(
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Validate model, migration, observation, alias, and source integrity."""
    _emit(_service(registry).validate_registry())


@registry_app.command("stale")
def registry_stale(
    as_of: Annotated[
        str, typer.Option("--as-of", help="Fixed ISO comparison date.")
    ] = date.today().isoformat(),
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Report local freshness metadata without refreshing it."""
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError as exc:
        typer.echo("--as-of must use YYYY-MM-DD format.", err=True)
        raise typer.Exit(2) from exc
    _emit(_service(registry).check_registry_freshness(as_of_date))


@registry_app.command("propose-update")
def registry_propose_update(
    research_result: Path,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    as_yaml: Annotated[bool, typer.Option("--yaml", help="Emit YAML.")] = False,
    output: Annotated[
        Path | None,
        typer.Option(help="Write review artifacts beneath this directory."),
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Create a review-only proposal; never modify canonical registry files."""
    if as_json and as_yaml:
        typer.echo("Choose only one of --json or --yaml.", err=True)
        raise typer.Exit(2)
    research = _read_research(research_result)
    service = _service(registry)
    if output:
        proposal, target = service.write_registry_proposal(research, output)
        typer.echo(f"Wrote review artifacts to {target}", err=True)
    else:
        proposal = service.propose_registry_update(research)
    if as_json:
        _emit(proposal)
    elif as_yaml:
        typer.echo(proposal_as_yaml(proposal), nl=False)
    else:
        typer.echo(proposal_as_markdown(proposal), nl=False)


@models_app.command("list")
def list_models(
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """List canonical model identities as JSON."""
    identities = [
        profile.identity.model_dump(mode="json") for profile in _service(registry).registry.all()
    ]
    _emit(identities)


@models_app.command("resolve")
def resolve(
    identifier: str,
    platform: Annotated[str | None, typer.Option()] = None,
    endpoint: Annotated[str | None, typer.Option()] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Resolve an identifier and expose its effective platform capabilities."""
    try:
        resolution = _service(registry).resolve_model(identifier, platform, endpoint)
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    payload = resolution.model_dump(mode="json")
    payload["canonical_name"] = resolution.canonical_name
    _emit(payload)


@models_app.command("match")
def match(
    identifier: str,
    platform: Annotated[str | None, typer.Option()] = None,
    endpoint: Annotated[str | None, typer.Option()] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Match a possibly vague identifier and suggest candidates instead of failing."""
    _emit(_service(registry).match_model(identifier, platform, endpoint))


@models_app.command("show")
def show(
    identifier: str,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Show a validated model profile."""
    try:
        _emit(_service(registry).get_model_profile(identifier))
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@models_app.command("lifecycle")
def model_lifecycle(
    identifier: str,
    as_of: Annotated[
        str, typer.Option("--as-of", help="Fixed ISO comparison date.")
    ] = date.today().isoformat(),
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Check whether reviewed lifecycle facts make a model suitable for migration."""
    try:
        as_of_date = date.fromisoformat(as_of)
        _emit(_service(registry).check_model_lifecycle(identifier, as_of_date=as_of_date))
    except ValueError as exc:
        typer.echo(f"Invalid lifecycle input: {exc}", err=True)
        raise typer.Exit(2) from exc
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
def compare(
    source: str,
    target: str,
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    source_endpoint: Annotated[str | None, typer.Option("--from-endpoint")] = None,
    target_endpoint: Annotated[str | None, typer.Option("--to-endpoint")] = None,
    live_pricing: Annotated[
        bool, typer.Option(help="Attach an opt-in OpenRouter pricing overlay.")
    ] = False,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Compare two model profiles."""
    try:
        _emit(
            _service(registry).compare_models(
                source,
                target,
                source_platform,
                target_platform,
                source_endpoint,
                target_endpoint,
                include_live_pricing=live_pricing,
            )
        )
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
def recommend(
    platform: Annotated[str | None, typer.Option()] = None,
    provider: Annotated[str | None, typer.Option()] = None,
    region: Annotated[str | None, typer.Option()] = None,
    require: Annotated[
        list[str] | None,
        typer.Option("--require", help="Required capability; repeatable."),
    ] = None,
    min_context: Annotated[int | None, typer.Option("--min-context")] = None,
    goal: Annotated[MigrationGoal, typer.Option()] = MigrationGoal.BALANCED,
    source: Annotated[str | None, typer.Option("--from")] = None,
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    application: Annotated[
        Path | None, typer.Option(help="Scan this Python application for requirements.")
    ] = None,
    live_pricing: Annotated[
        bool, typer.Option(help="Attach an opt-in OpenRouter pricing overlay.")
    ] = False,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Filter hard constraints, then rank compatible models deterministically."""
    constraints = RecommendationConstraints(
        platform=platform,
        provider=provider,
        region=region,
        required_capabilities=set(require or []),
        minimum_context_window=min_context,
        migration_goal=goal,
        source_model=source,
        source_platform=source_platform,
    )
    try:
        service = _service(registry)
        analysis = service.scan_application(application) if application else None
        _emit(service.recommend_models(constraints, analysis, include_live_pricing=live_pricing))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command("scan-application")
def scan_application_command(
    path: Path,
    prompt_source: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-source",
            help="Explicit prompt source file (relative to the application root); repeatable.",
        ),
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Scan a Python application into a provider-neutral coupling inventory."""
    try:
        _emit(_service(registry).scan_application(path, prompt_sources=prompt_source))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command("query-live-pricing")
def query_live_pricing_command(
    identifier: str,
    timeout: Annotated[float, typer.Option(min=0.1, max=30.0)] = 5.0,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Explicitly query OpenRouter-scoped pricing evidence; never update the registry."""
    try:
        _emit(_service(registry).query_live_pricing(identifier, timeout=timeout))
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command("estimate-cost")
def estimate_cost(
    source: Annotated[str, typer.Option("--from")],
    target: Annotated[str, typer.Option("--to")],
    requests: Annotated[int, typer.Option(min=1)],
    input_tokens: Annotated[int, typer.Option("--input-tokens", min=0)],
    output_tokens: Annotated[int, typer.Option("--output-tokens", min=0)],
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Estimate canonical recurring token cost for the same source/target workload."""
    try:
        workload = MigrationWorkload(
            requests=requests,
            input_tokens_per_request=input_tokens,
            output_tokens_per_request=output_tokens,
        )
        _emit(_service(registry).estimate_migration_cost(source, target, workload))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@prompt_app.command("analyze")
def prompt_analyze(path: Path) -> None:
    """Statically analyze a UTF-8 prompt file."""
    _emit(_service().analyze_prompt(_read_prompt(path)))


@prompt_app.command("prepare-migration")
def prompt_prepare_migration(
    path: Path,
    source: Annotated[str, typer.Option("--from")],
    target: Annotated[str, typer.Option("--to")],
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    source_role: Annotated[str, typer.Option("--role")] = "unknown",
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Prepare a structured prompt migration spec without rewriting the prompt."""
    try:
        if source_role not in {"system", "user", "developer", "unknown"}:
            raise ValueError("--role must be system, user, developer, or unknown")
        result = _service(registry).prepare_prompt_migration(
            source,
            target,
            _read_prompt(path),
            source_path=str(path),
            source_role=cast(Literal["system", "user", "developer", "unknown"], source_role),
            source_platform=source_platform,
            target_platform=target_platform,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(result)


@prompt_app.command("validate")
def prompt_validate(
    path: Path,
    target: Annotated[str, typer.Option("--to")],
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    target_endpoint: Annotated[str | None, typer.Option("--to-endpoint")] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Statically validate prompt assumptions against a target model."""
    try:
        result = _service(registry).validate_prompt(
            target,
            _read_prompt(path),
            source_path=str(path),
            target_platform=target_platform,
            target_endpoint=target_endpoint,
        )
    except RegistryError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(result)


@invocation_app.command("analyze")
def invocation_analyze(
    path: Path,
    target: Annotated[str | None, typer.Option("--to")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Normalize invocation, parameter, tool, and output couplings."""
    try:
        _emit(
            _service(registry).analyze_invocation(
                path, target=target, target_platform=target_platform
            )
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@invocation_app.command("prepare-migration")
def invocation_prepare_migration(
    path: Path,
    source: Annotated[str, typer.Option("--from")],
    target: Annotated[str, typer.Option("--to")],
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Prepare a review-only target invocation representation."""
    try:
        _emit(
            _service(registry).prepare_invocation_migration(
                path,
                source,
                target,
                source_platform=source_platform,
                target_platform=target_platform,
            )
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
def plan(
    application: Path,
    source: Annotated[str, typer.Option("--from")],
    target: Annotated[str, typer.Option("--to")],
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    session: Annotated[
        Path | None,
        typer.Option(help="Use the session registry overlay recorded in this run directory."),
    ] = None,
    session_as_of: Annotated[
        str | None,
        typer.Option(
            "--session-as-of",
            help="Fixed ISO timestamp for session-overlay expiry checks.",
        ),
    ] = None,
    prompt_source: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-source",
            help="Explicit prompt source file (relative to the application root); repeatable.",
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of YAML.")] = False,
    output: Annotated[
        Path | None,
        typer.Option(help="Write migration-manifest.yaml to this explicit path."),
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Generate an application-level migration manifest without editing the application."""
    try:
        service = _service(registry)
        service, session_lines = _apply_session_overlay(service, session, session_as_of)
        result = service.generate_migration_plan(
            application,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
            prompt_sources=prompt_source,
        )
        if session_lines:
            result = result.model_copy(update={"warnings": [*result.warnings, *session_lines]})
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    rendered = (
        result.model_dump_json(indent=2) + "\n"
        if as_json
        else service.migration_manifest_as_yaml(result)
    )
    if output:
        _write_migration_artifact(
            output,
            rendered,
            application,
            result.affected_files,
            "migration manifest",
        )
        typer.echo(f"Wrote migration manifest to {output}", err=True)
    else:
        typer.echo(rendered, nl=False)


@app.command("report")
def report(
    application: Path,
    source: Annotated[str, typer.Option("--from")],
    target: Annotated[str, typer.Option("--to")],
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    session: Annotated[
        Path | None,
        typer.Option(help="Use the session registry overlay recorded in this run directory."),
    ] = None,
    session_as_of: Annotated[
        str | None,
        typer.Option(
            "--session-as-of",
            help="Fixed ISO timestamp for session-overlay expiry checks.",
        ),
    ] = None,
    prompt_source: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-source",
            help="Explicit prompt source file (relative to the application root); repeatable.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write the Markdown report to this explicit path."),
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Generate the human review report for an integrated migration plan."""
    try:
        service = _service(registry)
        service, session_lines = _apply_session_overlay(service, session, session_as_of)
        plan = service.generate_migration_plan(
            application,
            source,
            target,
            source_platform=source_platform,
            target_platform=target_platform,
            prompt_sources=prompt_source,
        )
        if session_lines:
            plan = plan.model_copy(update={"warnings": [*plan.warnings, *session_lines]})
        rendered = service.migration_report(plan)
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_migration_artifact(
            output,
            rendered,
            application,
            plan.affected_files,
            "migration report",
        )
        typer.echo(f"Wrote migration report to {output}", err=True)
    else:
        typer.echo(rendered, nl=False)


@eval_app.command("generate-suite")
def eval_generate_suite(
    manifest: Path,
    corpus: Path,
    name: Annotated[str, typer.Option()] = "migration-evaluation",
    output: Annotated[
        Path | None, typer.Option(help="Write the generated suite YAML to this explicit path.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Bind a corpus to a v0.4 migration manifest without making model calls."""
    try:
        manifest_raw = _read_yaml(manifest, "migration manifest")
        if isinstance(manifest_raw, dict) and "migration" in manifest_raw:
            manifest_raw = manifest_raw["migration"]
        plan = MigrationPlan.model_validate(manifest_raw)
        corpus_raw = _read_yaml(corpus, "evaluation corpus")
        if isinstance(corpus_raw, dict):
            corpus_raw = corpus_raw.get("cases")
        if not isinstance(corpus_raw, list):
            raise ValueError("evaluation corpus must be a list or an object with a cases list")
        cases = [EvaluationCase.model_validate(item) for item in corpus_raw]
        service = _service(registry)
        suite = service.generate_eval_suite(plan, cases, name=name)
        rendered = service.evaluation_artifact_as_yaml(suite)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Invalid evaluation input: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_eval_artifact(output, rendered, [manifest, corpus], "evaluation suite")
        typer.echo(f"Wrote evaluation suite to {output}", err=True)
    else:
        typer.echo(rendered, nl=False)


@eval_app.command("run")
def eval_run(
    suite_path: Path,
    source_config_path: Path,
    target_config_path: Path,
    output: Annotated[
        Path | None, typer.Option(help="Persist the credential-free evaluation run YAML.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Run the same suite against source and target using local BYOK configuration."""
    try:
        suite = EvaluationSuite.model_validate(_read_yaml(suite_path, "evaluation suite"))
        source_config = EvaluationRunConfig.model_validate(
            _read_yaml(source_config_path, "source evaluation config")
        )
        target_config = EvaluationRunConfig.model_validate(
            _read_yaml(target_config_path, "target evaluation config")
        )
        service = _service(registry)
        run = service.run_migration_eval(suite, source_config, target_config)
        rendered = service.evaluation_artifact_as_yaml(run)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Invalid evaluation input: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_eval_artifact(
            output,
            rendered,
            [suite_path, source_config_path, target_config_path],
            "evaluation run",
        )
        typer.echo(f"Wrote evaluation run to {output}", err=True)
    else:
        typer.echo(rendered, nl=False)


@eval_app.command("compare-outputs")
def eval_compare_outputs(
    run_path: Path,
    case_id: Annotated[str | None, typer.Option(help="Compare only this case id.")] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Compute deterministic source/target deltas from a persisted run."""
    try:
        run = MigrationEvalRun.model_validate(_read_yaml(run_path, "evaluation run"))
        service = _service(registry)
        comparisons = [
            service.compare_outputs(source, target)
            for source, target in zip(run.source_results, run.target_results, strict=True)
            if case_id is None or source.case_id == case_id
        ]
        if case_id is not None and not comparisons:
            raise ValueError(f"evaluation case {case_id!r} was not found")
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Invalid evaluation run: {exc}", err=True)
        raise typer.Exit(2) from exc
    _emit(comparisons)


@eval_app.command("analyze-regressions")
def eval_analyze_regressions(
    run_path: Path,
    output: Annotated[
        Path | None, typer.Option(help="Persist the structured regression report YAML.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Categorize source/target quality, validity, latency, token, and cost regressions."""
    try:
        run = MigrationEvalRun.model_validate(_read_yaml(run_path, "evaluation run"))
        service = _service(registry)
        report = service.analyze_regressions(run)
        rendered = service.evaluation_artifact_as_yaml(report)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Invalid evaluation run: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_eval_artifact(output, rendered, [run_path], "regression report")
        typer.echo(f"Wrote regression report to {output}", err=True)
    else:
        typer.echo(rendered, nl=False)


@eval_app.command("optimize")
def eval_optimize(
    report_path: Path,
    candidate_run: Annotated[
        list[Path] | None,
        typer.Option("--candidate-run", help="A reviewed candidate evaluation run; repeatable."),
    ] = None,
    objective: Annotated[
        Literal["balanced", "quality", "cost", "latency"], typer.Option()
    ] = "balanced",
    max_candidates: Annotated[int, typer.Option(min=1, max=10)] = 3,
    max_runs: Annotated[int, typer.Option(min=0, max=20)] = 3,
    max_cost_usd: Annotated[float | None, typer.Option(min=0)] = None,
    minimum_pass_rate: Annotated[float, typer.Option(min=0, max=1)] = 1.0,
    allow_quality_regression: Annotated[bool, typer.Option()] = False,
    output: Annotated[
        Path | None, typer.Option(help="Persist the review-only optimization result YAML.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Recommend bounded changes and compare explicitly supplied candidate runs."""
    paths = candidate_run or []
    try:
        report = RegressionReport.model_validate(_read_yaml(report_path, "regression report"))
        runs: dict[str, MigrationEvalRun] = {}
        for path in paths:
            candidate_id = path.stem
            if candidate_id in runs:
                raise ValueError(f"duplicate candidate id {candidate_id!r}")
            runs[candidate_id] = MigrationEvalRun.model_validate(
                _read_yaml(path, "candidate evaluation run")
            )
        limits = OptimizationLimits(
            objective=objective,
            max_candidates=max_candidates,
            max_evaluation_runs=max_runs,
            max_estimated_cost_usd=Decimal(str(max_cost_usd)) if max_cost_usd is not None else None,
            minimum_target_pass_rate=minimum_pass_rate,
            allow_quality_regression=allow_quality_regression,
        )
        service = _service(registry)
        result = service.optimize_migration(report, runs, limits=limits)
        rendered = service.evaluation_artifact_as_yaml(result)
    except (ValidationError, ValueError) as exc:
        typer.echo(f"Invalid optimization input: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_eval_artifact(output, rendered, [report_path, *paths], "optimization result")
        typer.echo(f"Wrote optimization result to {output}", err=True)
    else:
        typer.echo(rendered, nl=False)


def _read_typed(path: Path, schema: type[Any], label: str) -> Any:
    try:
        return schema.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        typer.echo(f"Invalid {label} {path}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _write_stage_artifact(output: Path, artifact: Any, label: str) -> None:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(artifact.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    except OSError as exc:
        typer.echo(f"Unable to write {label} {output}: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Wrote {label} to {output}", err=True)


@research_app.command("create-request")
def research_create_request(
    application: Path,
    run_id: Annotated[str, typer.Option("--run-id")],
    source_provider: Annotated[str, typer.Option("--source-provider")],
    source_platform: Annotated[str, typer.Option("--source-platform")],
    source_model: Annotated[str, typer.Option("--source-model")],
    target_provider: Annotated[str, typer.Option("--target-provider")],
    target_platform: Annotated[str, typer.Option("--target-platform")],
    target_model: Annotated[str, typer.Option("--target-model")],
    source_endpoint: Annotated[str | None, typer.Option("--source-endpoint")] = None,
    target_endpoint: Annotated[str | None, typer.Option("--target-endpoint")] = None,
    as_of: Annotated[
        str, typer.Option("--as-of", help="Fixed ISO research date.")
    ] = date.today().isoformat(),
    topic: Annotated[
        list[str] | None,
        typer.Option("--topic", help="Requested research topic; repeatable."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Write request.yaml to this explicit path.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Scan locally and bound explicit research to missing or stale knowledge."""
    try:
        as_of_date = date.fromisoformat(as_of)
        topics = [ResearchTopic(value) for value in topic] if topic else None
        request = _service(registry).create_migration_research_request(
            application,
            run_id=run_id,
            source=ModelEndpointIdentity(
                provider=source_provider,
                platform=source_platform,
                model=source_model,
                endpoint=source_endpoint,
            ),
            target=ModelEndpointIdentity(
                provider=target_provider,
                platform=target_platform,
                model=target_model,
                endpoint=target_endpoint,
            ),
            as_of=as_of_date,
            topics=topics,
        )
    except (RegistryError, ValidationError, ValueError) as exc:
        typer.echo(f"Unable to create research request: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_stage_artifact(output, request, "research request")
    else:
        _emit(request)


@research_app.command("validate-result")
def research_validate_result(
    research_result: Path,
    request_path: Annotated[Path, typer.Argument(metavar="REQUEST")],
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Run the deterministic scope/policy gate over one research artifact."""
    research = _read_research(research_result)
    request = _read_typed(request_path, MigrationResearchRequest, "research request")
    problems = _service(registry).validate_research_result(research, request)
    _emit({"valid": not problems, "problems": problems})
    if problems:
        raise typer.Exit(1)


@research_app.command("validate-review")
def research_validate_review(
    review_path: Annotated[Path, typer.Argument(metavar="REVIEW")],
    research_result: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Check that an evidence review actually covers the research artifact."""
    review = _read_typed(review_path, EvidenceReview, "evidence review")
    research = _read_research(research_result)
    problems = _service(registry).validate_evidence_review(review, research)
    _emit({"valid": not problems, "problems": problems})
    if problems:
        raise typer.Exit(1)


@research_app.command("validate-artifact")
def research_validate_artifact(
    run_dir: Path,
    scope: str,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Validate one scope's research artifacts directly from the run workspace."""
    try:
        result = _service(registry).validate_research_artifact(run_dir, scope)
    except ValueError as exc:
        typer.echo(f"Unable to validate the research artifact: {exc}", err=True)
        raise typer.Exit(2) from exc
    _emit(result)
    if not result.valid:
        raise typer.Exit(1)


@research_app.command("consensus")
def research_consensus(
    research_result: Path,
    review_path: Annotated[Path, typer.Argument(metavar="REVIEW")],
    request_path: Annotated[Path, typer.Argument(metavar="REQUEST")],
    output: Annotated[
        Path | None, typer.Option(help="Write consensus YAML to this explicit path.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Deterministically combine one research result and its independent review."""
    research = _read_research(research_result)
    review = _read_typed(review_path, EvidenceReview, "evidence review")
    request = _read_typed(request_path, MigrationResearchRequest, "research request")
    try:
        consensus = _service(registry).build_research_consensus(research, review, request)
    except ValueError as exc:
        typer.echo(f"Consensus failed: {exc}", err=True)
        raise typer.Exit(2) from exc
    if output:
        _write_stage_artifact(output, consensus, "research consensus")
    else:
        _emit(consensus)


@research_app.command("build-session")
def research_build_session(
    run_dir: Path,
    now: Annotated[
        str | None,
        typer.Option("--now", help="Fixed ISO timestamp for reproducible manifests."),
    ] = None,
    ttl_days: Annotated[int, typer.Option("--ttl-days", min=1, max=90)] = 7,
    shadow: Annotated[
        list[str] | None,
        typer.Option("--shadow", help="Explicitly shadow this canonical model; repeatable."),
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Finalize a run: consensus, gates, and the immutable expiring overlay.

    Research and review artifacts must already exist in the run workspace
    (written by the agent host); no agent, network, or inference work happens
    here, and any stage still needing an agent fails closed.
    """
    from llm_migrate.core.orchestration import NullAgentRunner

    request = _read_typed(run_dir / "request.yaml", MigrationResearchRequest, "research request")
    if request.run_id != run_dir.name:
        typer.echo(
            f"Run directory {run_dir.name!r} does not match request run_id {request.run_id!r}.",
            err=True,
        )
        raise typer.Exit(2)
    moment = utc_moment(now) or datetime.now(UTC)
    try:
        outcome = _service(registry).run_agent_research(
            request,
            NullAgentRunner(),
            run_dir.parent,
            now=moment,
            overlay_ttl=timedelta(days=ttl_days),
            shadow_canonical=shadow,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(f"Unable to build the session overlay: {exc}", err=True)
        raise typer.Exit(2) from exc
    _emit(outcome)
    if outcome.manifest is None:
        raise typer.Exit(1)


@research_app.command("fetch-sources")
def research_fetch_sources(
    research_result: Path,
    timeout: Annotated[float, typer.Option(min=0.1, max=60.0)] = 5.0,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Explicitly refetch the sources one research artifact cites.

    Records reachability and a content hash for reviewer support; page content
    is never stored, and no other URL is ever contacted.
    """
    research = _read_research(research_result)
    records = _service(registry).refetch_research_sources(research, timeout=timeout)
    _emit(records)


@research_app.command("status")
def research_status(run_dir: Path) -> None:
    """Show the recorded orchestration run for a research workspace."""
    from llm_migrate.core.orchestration import OrchestrationRun

    run_path = run_dir / "run.yaml"
    if not run_path.is_file():
        typer.echo(f"No orchestration run recorded at {run_path}.", err=True)
        raise typer.Exit(2)
    _emit(_read_typed(run_path, OrchestrationRun, "orchestration run"))


@research_app.command("prompts")
def research_prompts(
    run_dir: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Render scope-isolated researcher/reviewer prompts from the run's request."""
    try:
        _emit(_service(registry).get_research_prompts(run_dir))
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("start")
def run_start(
    application: Path,
    source: Annotated[str, typer.Option("--from")],
    target: Annotated[str, typer.Option("--to")],
    source_platform: Annotated[str | None, typer.Option("--from-platform")] = None,
    target_platform: Annotated[str | None, typer.Option("--to-platform")] = None,
    source_endpoint: Annotated[str | None, typer.Option("--from-endpoint")] = None,
    target_endpoint: Annotated[str | None, typer.Option("--to-endpoint")] = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Run workspace directory (default .llm-migrate/runs/<run-id>)."),
    ] = None,
    as_of: Annotated[str | None, typer.Option("--as-of", help="Fixed ISO date.")] = None,
    prompt_source: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt-source",
            help="Explicit prompt source file (relative to the application root); repeatable.",
        ),
    ] = None,
    skip_research: Annotated[
        bool, typer.Option("--skip-research", help="Never write a research request.")
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Production mode: unknown evidence, missing invocation facts, "
            "incomplete coverage, and a missing validation disposition become "
            "blockers/violations.",
        ),
    ] = False,
    defer_prompt_candidates: Annotated[
        bool,
        typer.Option(
            "--defer-prompt-candidates",
            help="Start even though prompt candidates await confirmation; decide on the "
            "live run with `run add-prompt-source`.",
        ),
    ] = False,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Match models registry-first, create the run workspace, and report next steps.

    Prints `needs_confirmation` with the candidate list (and writes nothing)
    when prompt consumers exist but no prompt source resolved.
    """
    try:
        _emit(
            _service(registry).start_migration_run(
                application,
                source,
                target,
                source_platform=source_platform,
                target_platform=target_platform,
                source_endpoint=source_endpoint,
                target_endpoint=target_endpoint,
                run_id=run_id,
                output_dir=output_dir,
                as_of=date.fromisoformat(as_of) if as_of else None,
                research="skip" if skip_research else "auto",
                prompt_sources=prompt_source,
                strict=strict,
                defer_prompt_candidates=defer_prompt_candidates,
            )
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("tasks")
def run_tasks(
    run_dir: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """List the run's per-file prompt and file adaptation worklist."""
    try:
        _emit(_service(registry).list_adaptation_tasks(run_dir))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("blockers")
def run_blockers(
    run_dir: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Show every unresolved blocker with its question and evidence-backed options.

    Present each question with its options and evidence verbatim to the user,
    one blocker at a time, then record each answer with `run decide`.
    """
    try:
        _emit(_service(registry).get_blocker_resolutions(run_dir))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("decide")
def run_decide(
    run_dir: Path,
    blocker_id: Annotated[str, typer.Argument(help="Blocker id from `run blockers`.")],
    option_id: Annotated[str, typer.Argument(help="Option id from `run blockers`.")],
    rationale: Annotated[
        str,
        typer.Option(
            "--rationale",
            help="The user's own rationale (required for the accept option).",
        ),
    ] = "",
    decided_on: Annotated[
        str | None, typer.Option("--decided-on", help="Fixed ISO decision date.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Record the user's decision for one blocker (durable in decisions.yaml)."""
    try:
        result = _service(registry).record_blocker_decision(
            run_dir,
            blocker_id,
            option_id,
            rationale,
            decided_on=date.fromisoformat(decided_on) if decided_on else None,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(result)
    if not result.accepted:
        raise typer.Exit(1)


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    """Annotated changes from a YAML (or JSON) file holding a list of entries."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read annotations file {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"annotations file {path} must hold a list of annotated-change entries")
    return data


def _parse_disposition(value: str) -> GuidanceDisposition:
    """Parse one `--dispose` value: `<guidance-id>=<disposition>[:<note>]`."""
    guidance_id, separator, rest = value.partition("=")
    if not separator or not guidance_id.strip() or not rest.strip():
        raise ValueError(
            f"invalid --dispose value {value!r}; expected "
            "<guidance-id>=applied|not_applicable|declined[:<note>]"
        )
    disposition, _, note = rest.partition(":")
    if disposition not in ("applied", "not_applicable", "declined"):
        raise ValueError(
            f"invalid disposition {disposition!r} in --dispose value {value!r}; "
            "expected applied, not_applicable, or declined"
        )
    return GuidanceDisposition(
        guidance_id=guidance_id.strip(),
        disposition=disposition,  # type: ignore[arg-type]
        note=note.strip(),
    )


@run_app.command("submit-prompt")
def run_submit_prompt(
    run_dir: Path,
    source_path: Annotated[str, typer.Argument(help="Prompt path relative to the app root.")],
    rationale: Annotated[str, typer.Option("--rationale")],
    content: Annotated[
        Path | None,
        typer.Option("--content", help="File holding the adapted prompt."),
    ] = None,
    change: Annotated[
        list[str] | None,
        typer.Option("--change", help="One change description; repeatable."),
    ] = None,
    dispose: Annotated[
        list[str] | None,
        typer.Option(
            "--dispose",
            help=(
                "One guidance disposition as "
                "<guidance-id>=applied|not_applicable|declined[:<note>]; repeatable. "
                "Every guidance item of the prompt's task must be disposed."
            ),
        ),
    ] = None,
    annotations: Annotated[
        Path | None,
        typer.Option(
            "--annotations",
            help=(
                "YAML file holding the list of annotated changes (operation, "
                "original_anchor/adapted_anchor, why, evidence); required for a "
                "changed submission."
            ),
        ),
    ] = None,
    allow_restructure: Annotated[
        bool,
        typer.Option(
            "--allow-restructure",
            help="Accept an intentional, justified restructure of the prompt's sections.",
        ),
    ] = False,
    unchanged: Annotated[
        bool,
        typer.Option(
            "--unchanged",
            help="Record that the prompt was reviewed and needs no change (no --content).",
        ),
    ] = False,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Validate and store one adapted prompt beneath the run's output/prompts/."""
    if content is None and not unchanged:
        typer.echo("--content is required unless --unchanged is set", err=True)
        raise typer.Exit(2)
    if content is not None and unchanged:
        typer.echo("--content and --unchanged are mutually exclusive", err=True)
        raise typer.Exit(2)
    try:
        result = _service(registry).submit_adapted_prompt(
            run_dir,
            source_path,
            _read_prompt(content) if content is not None else "",
            rationale,
            change or [],
            allow_restructure=allow_restructure,
            unchanged=unchanged,
            guidance_dispositions=[_parse_disposition(item) for item in dispose or []],
            annotated_changes=_load_annotations(annotations) if annotations else None,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(result)
    if not result.accepted:
        raise typer.Exit(1)


@run_app.command("submit-file")
def run_submit_file(
    run_dir: Path,
    source_path: Annotated[str, typer.Argument(help="File path relative to the app root.")],
    rationale: Annotated[str, typer.Option("--rationale")],
    content: Annotated[
        Path | None,
        typer.Option("--content", help="File holding the adapted content."),
    ] = None,
    change: Annotated[
        list[str] | None,
        typer.Option("--change", help="One change description; repeatable."),
    ] = None,
    dispose: Annotated[
        list[str] | None,
        typer.Option(
            "--dispose",
            help=(
                "One required-change disposition as "
                "<id>=applied|not_applicable|declined[:<note>]; repeatable. "
                "Every required change of the file's task must be disposed."
            ),
        ),
    ] = None,
    annotations: Annotated[
        Path | None,
        typer.Option(
            "--annotations",
            help=(
                "YAML file holding the list of annotated changes (operation, "
                "original_anchor/adapted_anchor, why, evidence); required for a "
                "changed submission of an existing file."
            ),
        ),
    ] = None,
    new_file: Annotated[
        bool, typer.Option("--new-file", help="The migration introduces this file.")
    ] = False,
    unchanged: Annotated[
        bool,
        typer.Option(
            "--unchanged",
            help="Record that the file was reviewed and needs no change (no --content).",
        ),
    ] = False,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Check and store one adapted application file beneath the run's output/files/."""
    if content is None and not unchanged:
        typer.echo("--content is required unless --unchanged is set", err=True)
        raise typer.Exit(2)
    if content is not None and unchanged:
        typer.echo("--content and --unchanged are mutually exclusive", err=True)
        raise typer.Exit(2)
    try:
        result = _service(registry).submit_adapted_file(
            run_dir,
            source_path,
            _read_prompt(content) if content is not None else "",
            rationale,
            change or [],
            new_file=new_file,
            unchanged=unchanged,
            guidance_dispositions=[_parse_disposition(item) for item in dispose or []],
            annotated_changes=_load_annotations(annotations) if annotations else None,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(result)
    if not result.accepted:
        raise typer.Exit(1)


@run_app.command("record-validation")
def run_record_validation(
    run_dir: Path,
    method: Annotated[
        str,
        typer.Argument(
            metavar="METHOD",
            help="byok_evaluation | generated_tests | accepted_without_validation",
        ),
    ],
    rationale: Annotated[
        str, typer.Option("--rationale", help="The user's own validation rationale.")
    ] = "",
    outcome: Annotated[
        str | None,
        typer.Option(
            "--outcome",
            help="generated_tests only: run_passed | run_failed | not_run (the user's run).",
        ),
    ] = None,
    outcome_summary: Annotated[
        str,
        typer.Option("--outcome-summary", help="generated_tests only: the test summary line."),
    ] = "",
    evaluation_run: Annotated[
        str | None,
        typer.Option(
            "--evaluation-run",
            help="byok_evaluation only: the MigrationEvalRun YAML for the finalized manifest.",
        ),
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Record how this run's migration was validated (durable per run)."""
    try:
        _emit(
            _service(registry).record_validation_disposition(
                run_dir,
                method,  # type: ignore[arg-type]
                rationale,
                outcome=outcome,  # type: ignore[arg-type]
                outcome_summary=outcome_summary,
                evaluation_run_path=evaluation_run,
            )
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("status")
def run_status(
    run_dir: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Show the run's state machine position and its single next action."""
    try:
        _emit(_service(registry).get_run_status(run_dir))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("confirm-unaffected")
def run_confirm_unaffected(
    run_dir: Path,
    paths: Annotated[list[str], typer.Argument(metavar="PATH...")],
    rationale: Annotated[
        str, typer.Option("--rationale", help="Why these files were reviewed as unaffected.")
    ],
    acknowledge_source_references: Annotated[
        bool,
        typer.Option(
            "--acknowledge-source-references",
            help="Also close sweep-flagged files whose source-model mention is intentional.",
        ),
    ] = False,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Record reviewed no-change entries for the worklist's unaffected files."""
    try:
        confirmation = _service(registry).confirm_unaffected(
            run_dir,
            paths,
            rationale,
            acknowledge_source_references=acknowledge_source_references,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(confirmation)
    if confirmation.confirmed != len(confirmation.results):
        raise typer.Exit(1)


@run_app.command("add-prompt-source")
def run_add_prompt_source(
    run_dir: Path,
    paths: Annotated[list[str] | None, typer.Argument(metavar="[PATH...]")] = None,
    dismiss: Annotated[
        list[str] | None,
        typer.Option(
            "--dismiss",
            help="Candidate file or consumer path:line the user says is not a prompt; repeatable.",
        ),
    ] = None,
    rationale: Annotated[
        str, typer.Option("--rationale", help="The user's reason (required with --dismiss).")
    ] = "",
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Add prompt sources to a live run (or dismiss candidates); the worklist re-derives."""
    try:
        update = _service(registry).add_prompt_sources(
            run_dir, paths or [], dismiss=dismiss or [], rationale=rationale
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(update)
    if not update.accepted:
        raise typer.Exit(1)


@run_app.command("confirm-consumer")
def run_confirm_consumer(
    run_dir: Path,
    location: Annotated[str, typer.Argument(metavar="PATH:LINE")],
    source_path: Annotated[str, typer.Argument(metavar="PROMPT_FILE")],
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Record that one dynamic prompt consumer reads one prompt file."""
    try:
        update = _service(registry).confirm_prompt_consumer(run_dir, location, source_path)
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(update)
    if not update.accepted:
        raise typer.Exit(1)


@run_app.command("finalize")
def run_finalize(
    run_dir: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Write the run's manifest, report with per-file changes/rationale, and gaps."""
    try:
        _emit(_service(registry).finalize_migration_run(run_dir))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("review")
def run_review(
    run_dir: Path,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Show every annotated change with its diff context and decision status.

    Present each pending change verbatim — why, evidence, before/after spans —
    one at a time, then record each user decision with `run decide-change`.
    """
    try:
        _emit(_service(registry).get_change_review(run_dir))
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@run_app.command("decide-change")
def run_decide_change(
    run_dir: Path,
    source_path: Annotated[str, typer.Argument(help="Deliverable path from `run review`.")],
    change_id: Annotated[str, typer.Argument(help="Change id from `run review`.")],
    decision: Annotated[str, typer.Argument(help="accepted or rejected.")],
    note: Annotated[
        str, typer.Option("--note", help="The user's reasoning for the decision.")
    ] = "",
    decided_on: Annotated[
        str | None, typer.Option("--decided-on", help="Fixed ISO decision date.")
    ] = None,
    registry: Annotated[Path | None, typer.Option(help="Registry root.")] = None,
) -> None:
    """Record one accept/reject decision; the deliverable is regenerated.

    Decisions are durable in change-decisions.yaml, keyed to the submitted
    content; rejections deterministically revert their edits in the
    deliverable under output/.
    """
    try:
        result = _service(registry).record_change_decision(
            run_dir,
            source_path,
            change_id,
            decision,
            note,
            decided_on=date.fromisoformat(decided_on) if decided_on else None,
        )
    except (RegistryError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    _emit(result)
    if not result.accepted:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
