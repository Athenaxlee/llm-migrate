"""V1.4 interactive blocker resolution: questions, options, durable decisions."""

from __future__ import annotations

import shutil
import textwrap
from datetime import date
from pathlib import Path

import pytest
import yaml

from llm_migrate.core.blockers import load_decision_log
from llm_migrate.core.models import ResolutionKind
from llm_migrate.core.workspace import load_run_config
from llm_migrate.service import MigrationService
from tests.unit.adaptation_helpers import dispose_all, edit_change

AS_OF = date(2026, 9, 18)

NATIVE_APP = textwrap.dedent(
    """\
    import json

    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "extract"}],
        max_tokens=500,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {}},
            }
        },
    )
    payload = json.loads(response.content[0].text)
    """
)


@pytest.fixture
def native_app(tmp_path: Path) -> Path:
    """An application using native structured output on the Anthropic API."""
    app = tmp_path / "native_app"
    app.mkdir()
    (app / "app.py").write_text(NATIVE_APP, encoding="utf-8")
    return app


def _start_blocked_run(service: MigrationService, app: Path) -> Path:
    """Target claude-sonnet-5 on Bedrock, whose override declares no structured output.

    The selector-qualified target spelling seeds the invocation selector (the
    bare Bedrock id is not invocable on demand).
    """
    start = service.start_migration_run(
        app,
        "claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="amazon-bedrock",
        target_endpoint="bedrock-runtime",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    return Path(start.paths.run_dir)


def _blocker_id(service: MigrationService, run_dir: Path, code: str) -> str:
    resolutions = service.get_blocker_resolutions(run_dir)
    return next(item.blocker.id for item in resolutions.resolutions if item.blocker.code == code)


def test_blocked_run_produces_questions_with_evidence_backed_options(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers, "the run must be blocked for this scenario"
    assert any("get_blocker_resolutions" in line for line in tasks.guidance)
    assert any("never retry submissions" in line.casefold() for line in tasks.guidance)

    resolutions = service.get_blocker_resolutions(run_dir)
    assert resolutions.run_id == load_run_config(run_dir).run_id
    assert {item.blocker.code for item in resolutions.resolutions} == {
        "structured_output_unsupported",
        "structured_output_payload_unsupported",
        "structured_output_mapping_missing",
    }
    assert any("VERBATIM" in line for line in resolutions.guidance)
    for item in resolutions.resolutions:
        assert item.question.endswith("?")
        assert 2 <= len(item.options) <= 5
        accept = item.options[-1]
        assert accept.kind is ResolutionKind.ACCEPT_WITH_RATIONALE
        assert accept.id == "accept"
        assert any("never a default" in text for text in accept.consequences)
        for option in item.options:
            assert option.next_step.tool == "record_blocker_decision"
            assert option.next_step.arguments["blocker_id"] == item.blocker.id
            assert option.next_step.arguments["option_id"] == option.id

    unsupported = next(
        item
        for item in resolutions.resolutions
        if item.blocker.code == "structured_output_unsupported"
    )
    # The blocker itself cites the platform-scoped registry fact (AWS card).
    assert any("docs.aws.amazon.com" in url for url in unsupported.blocker.evidence_urls)
    option_ids = [option.id for option in unsupported.options]
    assert "retarget-anthropic-api" in option_ids
    assert "redesign" in option_ids
    # An alternative model on Bedrock cannot clear the missing deterministic
    # structured-output mapping, so no alternative option may be offered here.
    assert not any(option_id.startswith("alternative-") for option_id in option_ids)
    retarget = next(o for o in unsupported.options if o.id == "retarget-anthropic-api")
    assert retarget.evidence_urls and retarget.evidence_urls[0].startswith("https://")
    assert retarget.target_change is not None
    assert retarget.target_change.platform == "anthropic-api"
    redesign = next(o for o in unsupported.options if o.id == "redesign")
    assert redesign.task is not None
    assert redesign.task.category == "output_contract"
    assert any("json.loads" in line for line in redesign.task.guidance)
    assert redesign.evidence_urls, "redesign guidance must cite registry evidence"


def test_retarget_decision_updates_run_identity_and_unblocks(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(
        run_dir,
        blocker_id,
        "retarget-anthropic-api",
        decided_on=AS_OF,
    )
    assert result.accepted, result.problems
    assert result.run_config_updated
    config = load_run_config(run_dir)
    assert config.target.platform == "anthropic-api"
    assert config.target_model_id == "claude-sonnet-5"
    # Retargeting to a structured-output-capable endpoint clears every
    # structured-output blocker at once.
    assert result.unresolved_blockers == []
    assert (run_dir / "decisions.yaml").is_file()
    log = load_decision_log(run_dir, config.run_id)
    assert [item.kind for item in log.decisions] == [ResolutionKind.RETARGET]

    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity != "blocked"
    assert finalization.unresolved_blockers == []
    assert len(finalization.resolved_blockers) == 1
    assert finalization.stale_decisions == []
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "## Decisions" in report
    assert "**Retarget**" in report


def test_redesign_decision_injects_required_task_with_evidence(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(run_dir, blocker_id, "redesign", decided_on=AS_OF)
    assert result.accepted, result.problems
    assert not result.run_config_updated
    # Only the decided blocker is suppressed; the sibling blockers stay live.
    assert len(result.unresolved_blockers) == 2
    assert not any(blocker_id in line for line in result.unresolved_blockers)

    tasks = service.list_adaptation_tasks(run_dir)
    app_task = next(task for task in tasks.file_tasks if task.source_path == "app.py")
    redesign_changes = [
        change.text
        for change in app_task.required_changes
        if "Redesign off native structured output" in change.text
    ]
    assert redesign_changes
    assert "Guidance:" in redesign_changes[0]
    assert "evidence:" in redesign_changes[0]
    # Fail closed: the remaining blockers keep the plan blocked.
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity == "blocked"
    assert len(finalization.unresolved_blockers) == 2
    assert len(finalization.resolved_blockers) == 1


def test_accept_requires_user_rationale_and_stays_reported(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    resolutions = service.get_blocker_resolutions(run_dir)
    blocker_ids = [item.blocker.id for item in resolutions.resolutions]

    refused = service.record_blocker_decision(run_dir, blocker_ids[0], "accept")
    assert not refused.accepted
    assert any("rationale" in problem for problem in refused.problems)
    assert not (run_dir / "decisions.yaml").is_file()

    rationale = "We ship without native structured output until Bedrock supports it."
    for blocker_id in blocker_ids:
        result = service.record_blocker_decision(
            run_dir, blocker_id, "accept", rationale, decided_on=AS_OF
        )
        assert result.accepted, result.problems
    assert result.unresolved_blockers == []

    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity != "blocked"
    assert finalization.unresolved_blockers == []
    assert len(finalization.resolved_blockers) == 3
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "**ACCEPTED RISK**" in report
    assert rationale in report
    # The manifest records the decisions durably alongside the plan.
    manifest = yaml.safe_load(Path(finalization.manifest_path).read_text(encoding="utf-8"))
    decisions = manifest["migration"]["decisions"]
    assert len(decisions) == 3
    assert all(item["decision"]["rationale"] == rationale for item in decisions)
    assert manifest["migration"]["blockers"] == []


def test_wrong_ids_fail_closed(service: MigrationService, native_app: Path) -> None:
    run_dir = _start_blocked_run(service, native_app)
    unknown = service.record_blocker_decision(run_dir, "nope:0000000000", "accept", "why")
    assert not unknown.accepted
    assert any("not an unresolved blocker" in problem for problem in unknown.problems)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    bad_option = service.record_blocker_decision(run_dir, blocker_id, "invent-something")
    assert not bad_option.accepted
    assert any("valid options" in problem for problem in bad_option.problems)
    assert not (run_dir / "decisions.yaml").is_file()


def test_consistency_blocker_resolved_by_source_correction(
    service: MigrationService, project_root: Path, tmp_path: Path
) -> None:
    app = tmp_path / "openai_app"
    shutil.copytree(project_root / "tests/fixtures/applications/openai_app", app)
    start = service.start_migration_run(
        app,
        "claude-sonnet-5",  # wrong: the code actually uses gpt-5.6-sol
        "claude-sonnet-4-6",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    run_dir = Path(start.paths.run_dir)
    resolutions = service.get_blocker_resolutions(run_dir)
    codes = {item.blocker.code for item in resolutions.resolutions}
    assert {
        "source_model_mismatch",
        "source_provider_mismatch",
        "source_platform_mismatch",
    } <= codes

    model_mismatch = next(
        item for item in resolutions.resolutions if item.blocker.code == "source_model_mismatch"
    )
    correction = next(
        option
        for option in model_mismatch.options
        if option.id == "correct-source-model-gpt-5.6-sol"
    )
    assert correction.kind is ResolutionKind.CORRECTION
    assert correction.source_change is not None
    # The correction pairs the detected model with its detected platform.
    assert correction.source_change.platform == "openai-api"
    assert any("start_migration" in text for text in correction.consequences)

    # A provider mismatch alone has no registry-backed correction of its own.
    provider_mismatch = next(
        item for item in resolutions.resolutions if item.blocker.code == "source_provider_mismatch"
    )
    assert [option.id for option in provider_mismatch.options] == ["accept"]
    assert provider_mismatch.no_registry_backed_option is not None

    result = service.record_blocker_decision(
        run_dir, model_mismatch.blocker.id, correction.id, decided_on=AS_OF
    )
    assert result.accepted, result.problems
    assert result.run_config_updated
    config = load_run_config(run_dir)
    assert config.source.model == "gpt-5.6-sol"
    assert config.source.platform == "openai-api"
    assert config.source_model_id == "gpt-5.6-sol"
    # Correcting the source clears every consistency blocker at once.
    assert not any("mismatch" in line for line in result.unresolved_blockers)


def test_stale_decision_is_reported_not_applied(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(run_dir, blocker_id, "redesign", decided_on=AS_OF)
    assert result.accepted
    # The application changes: native structured output is removed entirely.
    (native_app / "app.py").write_text(
        NATIVE_APP.replace(
            """    output_config={
        "format": {
            "type": "json_schema",
            "schema": {"type": "object", "properties": {}},
        }
    },
""",
            "",
        ),
        encoding="utf-8",
    )
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers == []
    app_task = next(task for task in tasks.file_tasks if task.source_path == "app.py")
    assert not any(
        "Redesign off native structured output" in change for change in app_task.required_changes
    ), "a stale decision must not inject its task"
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.unresolved_blockers == []
    assert finalization.resolved_blockers == []
    assert len(finalization.stale_decisions) == 1
    assert "STALE" in finalization.message
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "**STALE decision**" in report


def test_guided_run_drives_blocked_to_finalized_without_relisting(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers
    blocker_id = _blocker_id(service, run_dir, "structured_output_mapping_missing")
    decided = service.record_blocker_decision(
        run_dir, blocker_id, "retarget-anthropic-api", decided_on=AS_OF
    )
    assert decided.accepted and decided.unresolved_blockers == []
    # With zero unresolved blockers the workflow moves on; it never asks for
    # another blocker-resolution round.
    assert not any("get_blocker_resolutions" in step for step in decided.next_steps)
    assert any("list_adaptation_tasks" in step for step in decided.next_steps)

    tasks = service.list_adaptation_tasks(run_dir)
    assert tasks.blockers == []
    assert [task.source_path for task in tasks.file_tasks] == ["app.py"]
    submitted = service.submit_adapted_file(
        run_dir,
        "app.py",
        NATIVE_APP.replace("claude-sonnet-4-6", "claude-sonnet-5"),
        "Move the invocation to claude-sonnet-5 on the Anthropic API per the decision.",
        ["Replaced the model identifier with claude-sonnet-5."],
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(service, run_dir, "app.py"),
        annotated_changes=[
            edit_change(
                "claude-sonnet-4-6",
                "claude-sonnet-5",
                why="The recorded retarget decision moves the app to claude-sonnet-5.",
            )
        ],
    )
    assert submitted.accepted, submitted.message

    finalization = service.finalize_migration_run(run_dir)
    assert finalization.migration_complexity != "blocked"
    assert finalization.unresolved_blockers == []
    assert finalization.coverage_gaps == []
    assert len(finalization.resolved_blockers) == 1
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "## Decisions" in report
    assert "Blocker decisions: 1 applied (0 accepted risk), 0 superseded, 0 stale" in report


def test_mcp_blocker_tools_wrap_the_shared_workflow(
    service: MigrationService, native_app: Path
) -> None:
    from llm_migrate.mcp.server import (
        get_blocker_resolutions as mcp_get_blocker_resolutions,
    )
    from llm_migrate.mcp.server import (
        record_blocker_decision as mcp_record_blocker_decision,
    )

    run_dir = _start_blocked_run(service, native_app)
    payload = mcp_get_blocker_resolutions(str(run_dir))
    assert payload["resolutions"]
    first = payload["resolutions"][0]
    recorded = mcp_record_blocker_decision(
        str(run_dir),
        first["blocker"]["id"],
        "accept",
        rationale="Reviewed with the user; accepted for a staged rollout.",
    )
    assert recorded["accepted"] is True
    assert len(recorded["unresolved_blockers"]) == 2


def test_cli_blockers_and_decide_commands(service: MigrationService, native_app: Path) -> None:
    import json

    from typer.testing import CliRunner

    from llm_migrate.cli.main import app

    runner = CliRunner()
    run_dir = _start_blocked_run(service, native_app)
    listed = runner.invoke(app, ["run", "blockers", str(run_dir)])
    assert listed.exit_code == 0, listed.output
    resolutions = json.loads(listed.stdout)["resolutions"]
    assert resolutions
    blocker_id = resolutions[0]["blocker"]["id"]

    refused = runner.invoke(app, ["run", "decide", str(run_dir), blocker_id, "accept"])
    assert refused.exit_code == 1, refused.output

    decided = runner.invoke(
        app,
        [
            "run",
            "decide",
            str(run_dir),
            blocker_id,
            "accept",
            "--rationale",
            "User accepted the gap for now.",
        ],
    )
    assert decided.exit_code == 0, decided.output
    assert json.loads(decided.stdout)["accepted"] is True


@pytest.fixture
def oversize_prompt_run(service: MigrationService, tmp_path: Path) -> Path:
    """A run blocked by context_window_exceeded (plus no_invocation_adapter)."""
    app = tmp_path / "oversize_app"
    app.mkdir()
    (app / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (app / "prompt.txt").write_text("answer carefully " * 15000, encoding="utf-8")
    start = service.start_migration_run(
        app,
        "fixture-alpha-large-v1",
        "fixture-gamma-cheap-v1",
        source_platform="fixture-api",
        target_platform="budget-platform",
        as_of=AS_OF,
        research="skip",
        prompt_sources=["prompt.txt"],
    )
    assert start.status == "ready" and start.paths is not None
    return Path(start.paths.run_dir)


def test_accepted_prompt_blocker_no_longer_rejects_submission(
    service: MigrationService, oversize_prompt_run: Path
) -> None:
    run_dir = oversize_prompt_run
    resolutions = service.get_blocker_resolutions(run_dir)
    codes = {item.blocker.code for item in resolutions.resolutions}
    assert "context_window_exceeded" in codes
    for item in resolutions.resolutions:
        result = service.record_blocker_decision(
            run_dir,
            item.blocker.id,
            "accept",
            "We ship the oversize prompt and monitor truncation in production.",
            decided_on=AS_OF,
        )
        assert result.accepted, result.problems
    assert result.unresolved_blockers == []

    submitted = service.submit_adapted_prompt(
        run_dir,
        "prompt.txt",
        "",
        "Reviewed; the user accepted the context-window risk explicitly.",
        unchanged=True,
        submitted_on=AS_OF,
        guidance_dispositions=dispose_all(
            service, run_dir, "prompt.txt", disposition="not_applicable"
        ),
    )
    assert submitted.accepted, submitted.message
    assert any(
        "explicitly accepted by a recorded blocker decision" in issue.message
        for issue in submitted.validation.issues
    )
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.unresolved_blockers == []
    assert "prompt.txt" not in finalization.coverage_gaps


def test_redesign_on_prompt_blocker_reaches_the_prompt_task(
    service: MigrationService, oversize_prompt_run: Path
) -> None:
    run_dir = oversize_prompt_run
    resolutions = service.get_blocker_resolutions(run_dir)
    context = next(
        item for item in resolutions.resolutions if item.blocker.code == "context_window_exceeded"
    )
    assert context.blocker.locations, "prompt blockers must carry their file"
    result = service.record_blocker_decision(
        run_dir, context.blocker.id, "redesign", decided_on=AS_OF
    )
    assert result.accepted, result.problems
    tasks = service.list_adaptation_tasks(run_dir)
    prompt_task = next(task for task in tasks.prompt_tasks if task.source_path == "prompt.txt")
    assert any(
        "Required change:" in item.text and "Reduce the prompt content" in item.text
        for item in prompt_task.guidance
    ), prompt_task.guidance


def test_correction_option_carries_a_concrete_endpoint(
    service: MigrationService, tmp_path: Path
) -> None:
    app = tmp_path / "bedrock_sonnet5_app"
    app.mkdir()
    (app / "app.py").write_text(
        textwrap.dedent(
            """\
            import boto3

            client = boto3.client("bedrock-runtime")
            client.converse(
                modelId="anthropic.claude-sonnet-5",
                messages=[{"role": "user", "content": [{"text": "hello"}]}],
            )
            """
        ),
        encoding="utf-8",
    )
    start = service.start_migration_run(
        app,
        "claude-sonnet-4-6",  # wrong: the code runs claude-sonnet-5 on Bedrock
        "gpt-5.6-sol",
        source_platform="anthropic-api",
        target_platform="openai-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    run_dir = Path(start.paths.run_dir)
    resolutions = service.get_blocker_resolutions(run_dir)
    mismatch = next(
        item for item in resolutions.resolutions if item.blocker.code == "source_model_mismatch"
    )
    correction = next(
        option for option in mismatch.options if option.id == "correct-source-model-claude-sonnet-5"
    )
    # The corrected identity must be concrete: a multi-endpoint platform
    # without an endpoint would make the tool's own option unrecordable.
    assert correction.source_change is not None
    assert correction.source_change.platform == "amazon-bedrock"
    assert correction.source_change.endpoint == "bedrock-runtime"
    result = service.record_blocker_decision(
        run_dir, mismatch.blocker.id, correction.id, decided_on=AS_OF
    )
    assert result.accepted, result.problems
    config = load_run_config(run_dir)
    assert config.source.model == "claude-sonnet-5"
    assert config.source.endpoint == "bedrock-runtime"


def test_same_invalid_schema_in_two_files_stays_two_blockers(
    service: MigrationService, tmp_path: Path
) -> None:
    app = tmp_path / "twin_schema_app"
    app.mkdir()
    module = textwrap.dedent(
        """\
        from anthropic import Anthropic

        client = Anthropic()
        client.messages.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "find"}],
            max_tokens=100,
            tools=[
                {
                    "name": "find_order",
                    "input_schema": {"type": "object", "properties": []},
                }
            ],
        )
        """
    )
    (app / "one.py").write_text(module, encoding="utf-8")
    (app / "two.py").write_text(module, encoding="utf-8")
    start = service.start_migration_run(
        app,
        "claude-sonnet-4-6",
        "claude-sonnet-5",
        source_platform="anthropic-api",
        target_platform="anthropic-api",
        as_of=AS_OF,
        research="skip",
    )
    assert start.status == "ready" and start.paths is not None
    tasks = service.list_adaptation_tasks(start.paths.run_dir)
    schema_blockers = [line for line in tasks.blockers if "invalid_json_schema" in line]
    assert len(schema_blockers) == 2
    assert any("one.py" in line for line in schema_blockers)
    assert any("two.py" in line for line in schema_blockers)


def test_foreign_decision_log_is_refused(service: MigrationService, native_app: Path) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(
        run_dir, blocker_id, "accept", "Reviewed and accepted.", decided_on=AS_OF
    )
    assert result.accepted
    log = yaml.safe_load((run_dir / "decisions.yaml").read_text(encoding="utf-8"))
    log["run_id"] = "some-other-run"
    (run_dir / "decisions.yaml").write_text(yaml.safe_dump(log), encoding="utf-8")
    from llm_migrate.core.workspace import WorkspaceError

    with pytest.raises(WorkspaceError, match="belongs to run 'some-other-run'"):
        service.list_adaptation_tasks(run_dir)


def test_placeholder_rationale_is_refused(service: MigrationService, native_app: Path) -> None:
    run_dir = _start_blocked_run(service, native_app)
    resolutions = service.get_blocker_resolutions(run_dir)
    first = resolutions.resolutions[0]
    accept = first.options[-1]
    # The advertised next_step never embeds a rationale an agent could echo.
    assert "rationale" not in accept.next_step.arguments
    refused = service.record_blocker_decision(
        run_dir,
        first.blocker.id,
        "accept",
        "<the user's own free-text rationale (required)>",
    )
    assert not refused.accepted
    assert any("placeholder" in problem for problem in refused.problems)


def test_superseded_retarget_is_history_not_a_stale_alarm(
    service: MigrationService, native_app: Path
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    blocker_id = _blocker_id(service, run_dir, "structured_output_unsupported")
    result = service.record_blocker_decision(
        run_dir, blocker_id, "retarget-anthropic-api", decided_on=AS_OF
    )
    assert result.accepted
    # Simulate an earlier retarget that the recorded one superseded.
    log = yaml.safe_load((run_dir / "decisions.yaml").read_text(encoding="utf-8"))
    earlier = dict(log["decisions"][0])
    earlier.update(
        {
            "blocker_id": "structured_output_unsupported:0000000000",
            "option_id": "retarget-amazon-bedrock-bedrock-mantle",
            "summary": "Keep claude-sonnet-5 but run it on bedrock-mantle.",
            "target_change": {
                "model": "claude-sonnet-5",
                "platform": "amazon-bedrock",
                "endpoint": "bedrock-mantle",
            },
        }
    )
    log["decisions"] = [earlier, *log["decisions"]]
    (run_dir / "decisions.yaml").write_text(yaml.safe_dump(log), encoding="utf-8")
    finalization = service.finalize_migration_run(run_dir)
    assert finalization.stale_decisions == []
    assert len(finalization.resolved_blockers) == 2
    assert any("superseded" in line for line in finalization.resolved_blockers)
    report = Path(finalization.report_path).read_text(encoding="utf-8")
    assert "**Superseded**" in report
    assert "STALE" not in report


def test_blocker_calls_scan_the_application_once(
    service: MigrationService, native_app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _start_blocked_run(service, native_app)
    calls = {"scan": 0}
    original = MigrationService.scan_application

    def counting(self, root, **kwargs):  # type: ignore[no-untyped-def]
        calls["scan"] += 1
        return original(self, root, **kwargs)

    monkeypatch.setattr(MigrationService, "scan_application", counting)
    resolutions = service.get_blocker_resolutions(run_dir)
    assert calls["scan"] == 1
    calls["scan"] = 0
    result = service.record_blocker_decision(
        run_dir,
        resolutions.resolutions[0].blocker.id,
        "accept",
        "Accepted during the efficiency check.",
        decided_on=AS_OF,
    )
    assert result.accepted
    assert calls["scan"] == 1
