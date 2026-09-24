"""V1.6.0-b: unknowns that give directions.

Typed `MigrationUnknown` records (plan v5) with subject / why it matters /
action / closing condition, exact run-scoped actions, BYOK probes for
contested registry facts, run-scoped observations, and the closing loop in
status and finalize.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import shlex
import shutil
from datetime import date
from pathlib import Path

import pytest

from llm_migrate.core.models import RUN_DIR_PLACEHOLDER, MigrationPlan, UnknownKind
from llm_migrate.core.observations import OBSERVATIONS_FILENAME, load_observations
from llm_migrate.core.workspace import load_run_config
from llm_migrate.mcp import server as mcp_server
from llm_migrate.service import MigrationService

AS_OF = date(2026, 9, 23)
BEDROCK = {
    "source_platform": "amazon-bedrock",
    "target_platform": "amazon-bedrock",
    "target_endpoint": "bedrock-runtime",
}
BEDROCK_PAIR = ("us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-5")
DYNAMIC_APP = """\
import anthropic

client = anthropic.Anthropic()
request = build_request()
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    tools=load_tools(),
    messages=[{"role": "user", "content": "x"}],
    **request,
)
"""


def _copy(project_root: Path, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name.split("/")[0]
    shutil.copytree(project_root / "tests/fixtures/applications" / name.split("/")[0], target)
    return tmp_path / name


@pytest.fixture
def field_app(tmp_path: Path, project_root: Path) -> Path:
    return _copy(project_root, tmp_path, "field_pattern_repo/app")


@pytest.fixture
def unresolved_field_app(field_app: Path) -> Path:
    config = field_app / "config/model_profiles.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(".yaml\n", "\n"), encoding="utf-8")
    return field_app


@pytest.fixture
def dynamic_app(tmp_path: Path) -> Path:
    app = tmp_path / "dynamic_app"
    app.mkdir()
    (app / "app.py").write_text(DYNAMIC_APP, encoding="utf-8")
    return app


def _bedrock_run(service: MigrationService, app: Path, **extra) -> Path:  # type: ignore[no-untyped-def]
    start = service.start_migration_run(
        app, *BEDROCK_PAIR, as_of=AS_OF, research="skip", **BEDROCK, **extra
    )
    assert start.status == "ready" and start.paths is not None, start.next_steps
    return Path(start.paths.run_dir)


def _run_plan(service: MigrationService, run_dir: Path) -> MigrationPlan:
    return service._plan_for_run(load_run_config(run_dir), run_dir)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(root).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _assert_action_is_exact(action: str, run_dir: Path) -> None:
    """The action is a valid MCP tool call with real parameters, or a CLI command."""
    if action.startswith("python "):
        argv = shlex.split(action)
        assert len(argv) == 2 and Path(argv[1]).is_file(), action
        assert Path(argv[1]).is_relative_to(run_dir)
        return
    call = ast.parse(action, mode="eval").body
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name), action
    tool = getattr(mcp_server, call.func.id)
    parameters = inspect.signature(tool).parameters
    names = {keyword.arg for keyword in call.keywords}
    assert names <= set(parameters), (action, names - set(parameters))
    required = {
        name for name, item in parameters.items() if item.default is inspect.Parameter.empty
    }
    assert required <= names, (action, required - names)
    arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    assert arguments["run_dir"] == str(run_dir.resolve())


# -- typed unknowns (plan v5) -------------------------------------------------


def test_every_emitter_populates_all_four_fields(
    service: MigrationService,
    project_root: Path,
    unresolved_field_app: Path,
    field_app: Path,
    dynamic_app: Path,
) -> None:
    fixtures = project_root / "tests/fixtures/applications"
    plans = [
        service.generate_migration_plan(fixtures / name, source, target, **platforms)
        for name, source, target, platforms in (
            ("anthropic_app", "claude-sonnet-4-6", "claude-sonnet-5", {}),
            ("openai_app", "gpt-5.6-sol", "claude-sonnet-5", {}),
            ("bedrock_app", "claude-sonnet-4-6", "claude-sonnet-5", BEDROCK),
            ("configured_prompt_app", "claude-sonnet-4-6", "claude-sonnet-5", BEDROCK),
            ("probe_forms_app", "claude-sonnet-4-6", "claude-sonnet-5", {}),
        )
    ]
    plans.append(service.generate_migration_plan(field_app, *BEDROCK_PAIR, **BEDROCK))
    plans.append(service.generate_migration_plan(unresolved_field_app, *BEDROCK_PAIR, **BEDROCK))
    plans.append(
        service.generate_migration_plan(
            dynamic_app,
            "claude-sonnet-4-6",
            "gpt-5.6-sol",
            source_platform="anthropic-api",
            target_platform="openai-api",
        )
    )
    kinds: set[UnknownKind] = set()
    statuses: set[str] = set()
    for plan in plans:
        assert plan.schema_version == "5"
        for unknown in plan.unknowns:
            kinds.add(unknown.kind)
            statuses.add(unknown.status)
            for text in (
                unknown.subject,
                unknown.why_it_matters,
                unknown.action,
                unknown.closing_condition,
                unknown.message,
            ):
                assert text.strip(), unknown
            assert (unknown.closed_reason is None) == (unknown.status == "open"), unknown
            # Outside a run, actions address the run through the placeholder.
            assert RUN_DIR_PLACEHOLDER in unknown.action, unknown.action
    assert kinds == set(UnknownKind), "every emitter must be exercised"
    assert statuses >= {"open", "closed_by_scan"}


def test_unknown_ids_are_stable_across_regenerations(
    service: MigrationService, field_app: Path
) -> None:
    first = service.generate_migration_plan(field_app, *BEDROCK_PAIR, **BEDROCK)
    second = service.generate_migration_plan(field_app, *BEDROCK_PAIR, **BEDROCK)
    assert [item.id for item in first.unknowns] == [item.id for item in second.unknowns]
    assert len({item.id for item in first.unknowns}) == len(first.unknowns)


def test_run_actions_are_exact_calls_for_that_run(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    run_dir = _bedrock_run(service, unresolved_field_app, defer_prompt_candidates=True)
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.unknowns, "the worklist carries open unknowns with their actions"
    plan = _run_plan(service, run_dir)
    kinds = {item.kind for item in plan.unknowns}
    assert {
        UnknownKind.PROMPT_CANDIDATE,
        UnknownKind.DYNAMIC_PROMPT_CONSUMER,
        UnknownKind.CONTESTED_EVIDENCE,
    } <= kinds
    for unknown in plan.unknowns:
        assert RUN_DIR_PLACEHOLDER not in unknown.action
        _assert_action_is_exact(unknown.action, run_dir)
    candidate = next(item for item in plan.unknowns if item.kind is UnknownKind.PROMPT_CANDIDATE)
    assert candidate.action.startswith("add_prompt_sources(")
    consumer = next(
        item for item in plan.unknowns if item.kind is UnknownKind.DYNAMIC_PROMPT_CONSUMER
    )
    assert consumer.action.startswith("confirm_prompt_consumer(")
    assert "invoke_" in (consumer.data or {})["location"]


def test_executing_a_candidate_action_closes_it(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    run_dir = _bedrock_run(service, unresolved_field_app, defer_prompt_candidates=True)
    plan = _run_plan(service, run_dir)
    candidate = next(
        item
        for item in plan.unknowns
        if item.kind is UnknownKind.PROMPT_CANDIDATE
        and "claude_prompt_multi" in (item.data or {})["path"]
    )
    call = ast.parse(candidate.action, mode="eval").body
    assert isinstance(call, ast.Call)
    arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    update = service.add_prompt_sources(**arguments)
    assert update.accepted, update.problems
    after = _run_plan(service, run_dir)
    assert candidate.id not in {item.id for item in after.unknowns if item.is_open}


# -- dismissals and confirmations close by action ---------------------------


def test_dismissals_and_confirmations_close_by_action(
    service: MigrationService, unresolved_field_app: Path
) -> None:
    run_dir = _bedrock_run(service, unresolved_field_app, defer_prompt_candidates=True)
    plan = _run_plan(service, run_dir)
    llama = sorted(
        str((item.data or {})["path"])
        for item in plan.unknowns
        if item.kind is UnknownKind.PROMPT_CANDIDATE and "llama" in str(item.data)
    )
    assert service.add_prompt_sources(
        run_dir, [], dismiss=llama, rationale="Llama profiles are retired."
    ).accepted

    def closed_by_action() -> dict[str, str]:
        return {
            str((item.data or {}).get("path") or (item.data or {}).get("location")): str(
                item.closed_reason
            )
            for item in _run_plan(service, run_dir).unknowns
            if item.status == "closed_by_action"
        }

    closed = closed_by_action()
    for path in llama:
        assert "Llama profiles are retired." in closed[path]
    assert service.confirm_prompt_consumer(
        run_dir,
        "analysis/invoke_multimodal.py:22:system",
        "analysis/prompt_lib/claude_prompt_multi.yaml",
    ).accepted
    closed = closed_by_action()
    assert "claude_prompt_multi.yaml" in closed["analysis/invoke_multimodal.py:22:system"]
    # With a selected sibling the llama files are out of scope; the user's
    # recorded dismissal is still what the plan shows for them.
    plan = _run_plan(service, run_dir)
    for path in llama:
        assert any(
            line.startswith(path) and "Llama profiles are retired." in line
            for line in plan.out_of_scope
        )
    finalization = service.finalize_migration_run(run_dir)
    assert any("invoke_multimodal.py:22" in item for item in finalization.unknowns_closed_by_action)
    assert finalization.unknowns_closed_by_scan  # parallel_tool_use: the app has no tools
    assert finalization.unknowns_open
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "## Closed unknowns" in report and "[closed by action]" in report
    assert "| Unknown | Why it matters | Action | Closes when |" in report


# -- contested evidence, probes, and observations ---------------------------


def test_probe_is_emitted_only_for_contested_evidence(
    service: MigrationService, field_app: Path, project_root: Path, tmp_path: Path
) -> None:
    run_dir = _bedrock_run(service, field_app)
    service.list_adaptation_tasks(run_dir)
    plan = _run_plan(service, run_dir)
    contested = [item for item in plan.unknowns if item.kind is UnknownKind.CONTESTED_EVIDENCE]
    assert len(contested) == 1
    unknown = contested[0]
    assert unknown.evidence_urls and len(unknown.evidence_urls) == 2
    assert "AWS model card" in unknown.why_it_matters
    probes = sorted(
        path.relative_to(run_dir).as_posix() for path in (run_dir / "output/probes").iterdir()
    )
    assert probes == [(unknown.data or {})["probe_path"]]
    source = (run_dir / probes[0]).read_text(encoding="utf-8")
    compile(source, probes[0], "exec")
    assert "additionalModelRequestFields=CONTESTED_SETTING" in source
    assert "'us.anthropic.claude-sonnet-5'" in source  # the run's invocation id
    assert unknown.id in source and "record_observation(" in source
    _assert_action_is_exact(unknown.action, run_dir)
    # A deleted probe makes the snapshot stale and is rewritten.
    (run_dir / probes[0]).unlink()
    assert service.get_run_status(run_dir).snapshot_reused is False
    assert (run_dir / probes[0]).is_file()

    # A conflict scoped to Bedrock does not govern an Anthropic API migration.
    app = tmp_path / "direct"
    shutil.copytree(project_root / "tests/fixtures/applications/anthropic_app", app)
    start = service.start_migration_run(
        app, "claude-sonnet-4-6", "claude-sonnet-5", as_of=AS_OF, research="skip"
    )
    assert start.paths is not None
    direct = Path(start.paths.run_dir)
    service.list_adaptation_tasks(direct)
    assert not (direct / "output/probes").exists()
    assert not any(
        item.kind is UnknownKind.CONTESTED_EVIDENCE for item in _run_plan(service, direct).unknowns
    )


def test_observation_closes_the_unknown_renders_and_never_touches_the_registry(
    service: MigrationService, field_app: Path, project_root: Path
) -> None:
    registry_before = _tree_digest(project_root / "registry")
    run_dir = _bedrock_run(service, field_app)
    plan = _run_plan(service, run_dir)
    contested = next(item for item in plan.unknowns if item.kind is UnknownKind.CONTESTED_EVIDENCE)
    before = service.get_run_status(run_dir).open_unknowns

    for subject, outcome, evidence, problem in (
        ("not-an-id", "REJECTED", "x", "not the id"),
        (contested.id, " ", "x", "outcome"),
        (contested.id, "REJECTED", "", "evidence"),
    ):
        refused = service.record_observation(run_dir, subject, outcome, evidence)
        assert not refused.accepted and any(problem in item for item in refused.problems)
    scanned = next(item for item in plan.unknowns if item.status == "closed_by_scan")
    assert not service.record_observation(run_dir, scanned.id, "x", "y").accepted
    assert not (run_dir / OBSERVATIONS_FILENAME).exists()

    result = service.record_observation(
        run_dir,
        contested.id,
        "REJECTED",
        "ValidationException: thinking cannot be disabled for this model",
    )
    assert result.accepted, result.problems
    assert service.get_run_status(run_dir).open_unknowns == before - 1
    after = next(item for item in _run_plan(service, run_dir).unknowns if item.id == contested.id)
    assert after.status == "closed_by_action" and "REJECTED" in str(after.closed_reason)
    # A later observation replaces the earlier one.
    service.record_observation(run_dir, contested.id, "ACCEPTED", "stopReason=end_turn")
    log = load_observations(run_dir, load_run_config(run_dir).run_id)
    assert [item.outcome for item in log.observations] == ["ACCEPTED"]

    finalization = service.finalize_migration_run(run_dir)
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "## Observations (run-scoped)" in report
    assert "stopReason=end_turn" in report
    assert _tree_digest(project_root / "registry") == registry_before


def test_mcp_and_cli_record_observations(
    service: MigrationService,
    field_app: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from llm_migrate.cli.main import app as cli_app

    run_dir = _bedrock_run(service, field_app)
    plan = _run_plan(service, run_dir)
    contested = next(item for item in plan.unknowns if item.kind is UnknownKind.CONTESTED_EVIDENCE)
    difference = next(
        item for item in plan.unknowns if item.kind is UnknownKind.MODEL_DIFFERENCE and item.is_open
    )
    monkeypatch.setenv("LLM_MIGRATE_REGISTRY", str(project_root / "registry"))
    result = mcp_server.record_observation(str(run_dir), contested.id, "REJECTED", "probe output")
    assert result["accepted"], result["problems"]
    runner = CliRunner()
    invoked = runner.invoke(
        cli_app,
        [
            "run",
            "record-observation",
            str(run_dir),
            difference.id,
            "--outcome",
            "Adaptive thinking is on by default; responses include reasoning.",
            "--evidence",
            "BYOK evaluation run 2026-09-23",
            "--registry",
            str(project_root / "registry"),
        ],
    )
    assert invoked.exit_code == 0, invoked.output
    refused = runner.invoke(
        cli_app,
        [
            "run",
            "record-observation",
            str(run_dir),
            "bogus",
            "--outcome",
            "x",
            "--evidence",
            "y",
            "--registry",
            str(project_root / "registry"),
        ],
    )
    assert refused.exit_code == 1


def test_probe_action_is_shell_quoted_for_awkward_run_dirs(
    service: MigrationService, tmp_path: Path, project_root: Path
) -> None:
    repo = tmp_path / "café run" / "field_pattern_repo"
    shutil.copytree(project_root / "tests/fixtures/applications/field_pattern_repo", repo)
    run_dir = _bedrock_run(service, repo / "app")
    service.list_adaptation_tasks(run_dir)
    plan = _run_plan(service, run_dir)
    contested = next(item for item in plan.unknowns if item.kind is UnknownKind.CONTESTED_EVIDENCE)
    argv = shlex.split(contested.action)
    assert argv[0] == "python" and Path(argv[1]).is_file(), contested.action
    assert "\\u00e9" not in contested.action
    # Tool-call actions stay JSON-escaped Python literals.
    consumer = next(
        item for item in plan.unknowns if item.kind is UnknownKind.DYNAMIC_PROMPT_CONSUMER
    )
    _assert_action_is_exact(consumer.action, run_dir)
